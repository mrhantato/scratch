/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/objective_function.h>

#include "symmetric_tree_learner.hpp"

namespace LightGBM {

SymmetricTreeLearner::SymmetricTreeLearner(const Config* config): 
SerialTreeLearner(config),
max_depth_(config->max_depth),
num_threads_(OMP_NUM_THREADS()) {
  if (max_depth_ <= 0) {
    Log::Fatal("To use symmetric_tree, please specify a positive max_depth.");
  }
  max_num_leaves_ = (1 << max_depth_);
}

void SymmetricTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();

  symmetric_data_partition_.reset(new SymmetricDataPartition(num_data_, max_num_leaves_, num_threads_));
  symmetric_histogram_pool_.reset(new SymmetricHistogramPool(num_threads_, max_num_leaves_));

  col_sampler_.SetTrainingData(train_data_);
  GetShareStates(train_data_, is_constant_hessian, true);
  symmetric_histogram_pool_->DynamicChangeSize(train_data_,
    share_state_->num_hist_total_bin(),
    share_state_->feature_hist_offsets(),
    config_, max_num_leaves_, max_num_leaves_);

  ordered_gradients_.resize(num_data_, 0.0f);
  ordered_hessians_.resize(num_data_, 0.0f);

  paired_leaf_indices_in_cur_level_.resize(max_num_leaves_);
  num_pairs_ = 0;

  thread_best_inner_feature_index_cur_level_.resize(num_threads_, -1);
  thread_best_threshold_cur_level_.resize(num_threads_, -1);
  thread_best_gain_cur_level_.resize(num_threads_, kMinScore);
  thread_best_split_default_left_cur_level_.resize(num_threads_, 0);

  leaf_indices_in_cur_level_.resize(max_num_leaves_, -1);

  best_level_split_info_.resize(max_num_leaves_);

  level_leaf_splits_.resize(max_num_leaves_);

  cur_level_ = 0;
  num_leaves_in_cur_level_ = 1;

  best_leaf_in_level_should_be_split_.resize(max_num_leaves_, 0);
}

Tree* SymmetricTreeLearner::Train(const score_t* gradients, const score_t *hessians, bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  BeforeTrain();
  // TODO(shiyu1994) support interaction constraints and linear tree
  std::unique_ptr<Tree> tree(new Tree(max_num_leaves_, false, false));
  for (int depth = 0; depth < config_->max_depth; ++depth) {
    SetUpLevelInfo(depth);
    PrepareLevelHistograms();
    // construct and subtract
    global_timer.Start("ConstructLevelHistograms");
    symmetric_data_partition_->ConstructLevelHistograms(&level_feature_histograms_, train_data_,
      col_sampler_.is_feature_used_bytree(),
      gradients, hessians, ordered_gradients_.data(), ordered_hessians_.data(),
      share_state_.get());
    global_timer.Stop("ConstructLevelHistograms");
    // find best splits
    global_timer.Start("FindBestLevelSplits");
    FindBestLevelSplits();
    global_timer.Stop("FindBestLevelSplits");
    const bool to_continue = SplitLevel(tree.get());
    if (!to_continue) {
      Log::Warning("No further splits found, stop training at level %d", depth);
      break;
    }
  }
  AfterTrain();
  return tree.release();
}

void SymmetricTreeLearner::PrepareLevelHistograms() {
  for (int i = 0; i < num_pairs_; ++i) {
    const std::vector<int>& pair = paired_leaf_indices_in_cur_level_[i];
    if (pair.size() == 2) {
      const int smaller_leaf_index = pair[0];
      const int larger_leaf_index = pair[1];
      if (smaller_leaf_index < larger_leaf_index) {
        symmetric_histogram_pool_->Move(smaller_leaf_index, larger_leaf_index);
      }
    }
  }
  for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
    const int leaf_id = leaf_indices_in_cur_level_[i];
    const bool get = symmetric_histogram_pool_->Get(leaf_id, &level_feature_histograms_[i]);
    if (!get) {
      // TODO(shiyu1994): handle the case when the feature histogram cache is not enough
      Log::Fatal("symmetric tree should have enough histograms in the pool");
    }
  }
}

void SymmetricTreeLearner::BeforeTrain() {
  leaf_indices_in_cur_level_[0] = 0;
  for (int i = 1; i < max_num_leaves_; ++i) {
    leaf_indices_in_cur_level_[i] = -1;
  }
  level_feature_histograms_.resize(max_num_leaves_, nullptr);

  // initalize leaf splits
  level_leaf_splits_[0].reset(new LeafSplits(num_data_, config_));
  level_leaf_splits_[0]->Init(gradients_, hessians_);

  paired_leaf_indices_in_cur_level_[0].resize(1);
  paired_leaf_indices_in_cur_level_[0][0] = 0;
  num_pairs_ = 1;
  num_leaves_in_cur_level_ = 1;

  col_sampler_.ResetByTree();

  symmetric_data_partition_->Init();
  symmetric_histogram_pool_->ResetMap();
}

void SymmetricTreeLearner::AfterTrain() {
  for (int i = 0; i < max_num_leaves_; ++i) {
    level_leaf_splits_[i].reset(nullptr);
  }
}

void SymmetricTreeLearner::FindBestLevelSplits() {
  std::vector<int8_t> thread_result_valid(num_threads_, 0);
  Threading::For<int>(0, num_features_, 1, [this, &thread_result_valid] (int thread_id, int start, int end) {
    for (int inner_feature_index = start; inner_feature_index < end; ++inner_feature_index) {
      const bool valid = FindBestLevelSplitsForFeature(inner_feature_index, thread_id);
      if (valid) {
        thread_result_valid[thread_id] = 1;
      }
    }
  });
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    if (thread_result_valid[thread_id] > 0 &&
        thread_best_gain_cur_level_[thread_id] > best_gain_cur_level_) {
      best_inner_feature_index_cur_level_ = thread_best_inner_feature_index_cur_level_[thread_id];
      best_threshold_cur_level_ = thread_best_threshold_cur_level_[thread_id];
      best_gain_cur_level_ = thread_best_gain_cur_level_[thread_id];
      best_split_default_left_cur_level_ = thread_best_split_default_left_cur_level_[thread_id];
    }
  }
  if (best_inner_feature_index_cur_level_ != -1) {
    #pragma omp parallel for schedule(static) num_threads(num_threads_) if (num_leaves_in_cur_level_ >= 1024)
    for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
      const int leaf_index = leaf_indices_in_cur_level_[i];
      SplitInfo& split_info = best_level_split_info_[i];
      symmetric_histogram_pool_->GetSplitLeafOutput(leaf_index, best_inner_feature_index_cur_level_,
        best_threshold_cur_level_, best_split_default_left_cur_level_, level_leaf_splits_[i],
        &best_leaf_in_level_should_be_split_, i,
        &split_info.left_output, &split_info.right_output,
        &split_info.left_sum_gradient, &split_info.left_sum_hessian, &split_info.right_sum_gradient, &split_info.right_sum_hessian,
        &split_info.left_count, &split_info.right_count, &split_info.gain);
    }
  }
}

void SymmetricTreeLearner::SetUpLevelInfo(const int depth) {
  cur_level_ = depth;
  best_inner_feature_index_cur_level_ = -1;
  best_threshold_cur_level_ = -1;
  best_gain_cur_level_ = kMinScore;
  best_split_default_left_cur_level_ = -1;

  for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
    best_leaf_in_level_should_be_split_[i] = 0;
  }
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_best_gain_cur_level_[thread_id] = kMinScore;
  }
  //TODO(shiyu1994) add level-wise feature subsampling
}

bool SymmetricTreeLearner::FindBestLevelSplitsForFeature(const int inner_feature_index, const int thread_id) {
  bool valid = symmetric_histogram_pool_->FindBestThresholdFromLevelHistograms(inner_feature_index,
    paired_leaf_indices_in_cur_level_,
    leaf_indices_in_cur_level_,
    level_leaf_splits_,
    &thread_best_inner_feature_index_cur_level_[thread_id],
    &thread_best_threshold_cur_level_[thread_id],
    &thread_best_gain_cur_level_[thread_id],
    &thread_best_split_default_left_cur_level_[thread_id],
    thread_id,
    num_leaves_in_cur_level_,
    train_data_,
    num_pairs_);
  return valid;
}

void SymmetricTreeLearner::CheckSplit(const int num_leaves_in_old_level,
    const std::vector<int>& old_left_child, const std::vector<int>& old_right_child,
    const std::vector<std::unique_ptr<LeafSplits>>& old_level_leaf_splits) {
  std::vector<std::vector<double>> thread_sum_gradients(num_threads_);
  std::vector<std::vector<double>> thread_sum_hessians(num_threads_);
  std::vector<std::vector<data_size_t>> thread_num_data(num_threads_);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_sum_gradients[thread_id].resize(num_leaves_in_cur_level_, 0.0f);
    thread_sum_hessians[thread_id].resize(num_leaves_in_cur_level_, 0.0f);
    thread_num_data[thread_id].resize(num_leaves_in_cur_level_, 0);
  }
  std::vector<double> sum_gradients(num_leaves_in_cur_level_, 0.0f);
  std::vector<double> sum_hessians(num_leaves_in_cur_level_, 0.0f);
  std::vector<data_size_t> num_datas(num_leaves_in_cur_level_, 0);
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, &thread_sum_gradients, &thread_sum_hessians, &thread_num_data]
      (int thread_id, data_size_t start, data_size_t end) {
      for (int i = start; i < end; ++i) {
        const int real_leaf_index = symmetric_data_partition_->GetDataLeafIndex(i);
        thread_sum_gradients[thread_id][real_leaf_index] += gradients_[i];
        thread_sum_hessians[thread_id][real_leaf_index] += hessians_[i];
        thread_num_data[thread_id][real_leaf_index] += 1;
      }
    });
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      sum_gradients[i] += thread_sum_gradients[thread_id][i];
      sum_hessians[i] += thread_sum_hessians[thread_id][i];
      num_datas[i] += thread_num_data[thread_id][i];
    }
  }
  for (int i = 0; i < num_leaves_in_old_level; ++i) {
    bool check_success = true;
    const SplitInfo& split_info = best_level_split_info_[i];
    const int left_real_leaf_index = old_left_child[i];
    const int right_real_leaf_index = old_right_child[i];
     if (right_real_leaf_index != -1) {
    if (split_info.left_sum_gradient != sum_gradients[left_real_leaf_index]) {
      check_success = false;
      Log::Warning("%f vs %f", split_info.left_sum_gradient, sum_gradients[left_real_leaf_index]);
    }
    //CHECK_EQ(split_info.left_sum_hessian, sum_hessians[left_real_leaf_index]);
    if (split_info.left_sum_hessian != sum_hessians[left_real_leaf_index]) {
      check_success = false;
      Log::Warning("%f vs %f", split_info.left_sum_hessian, sum_hessians[left_real_leaf_index]);
    }
    //CHECK_EQ(split_info.left_sum_hessian, sum_hessians[left_real_leaf_index]);
    if (split_info.left_count != num_datas[left_real_leaf_index]) {
      check_success = false;
      Log::Warning("%d vs %d", split_info.left_count, num_datas[left_real_leaf_index]);
    }
    //CHECK_EQ(split_info.left_count, num_datas[left_real_leaf_index]);
      if (split_info.right_sum_gradient != sum_gradients[right_real_leaf_index]) {
      check_success = false;
        Log::Warning("%f vs %f", split_info.right_sum_gradient, sum_gradients[right_real_leaf_index]);
      }
      //CHECK_EQ(split_info.right_sum_gradient, sum_gradients[right_real_leaf_index]);
      if (split_info.right_sum_hessian != sum_hessians[right_real_leaf_index]) {
      check_success = false;
        Log::Warning("%f vs %f", split_info.right_sum_hessian, sum_hessians[right_real_leaf_index]);
      }
      //CHECK_EQ(split_info.right_sum_hessian, sum_hessians[right_real_leaf_index]);
      if (split_info.right_count != num_datas[right_real_leaf_index]) {
      check_success = false;
        Log::Warning("%d vs %d", split_info.right_count, num_datas[right_real_leaf_index]);
      }
      //CHECK_EQ(split_info.right_count, num_datas[right_real_leaf_index]);
      const std::unique_ptr<LeafSplits>& leaf_splits = old_level_leaf_splits[i];
      if (split_info.left_sum_gradient + split_info.right_sum_gradient != leaf_splits->sum_gradients()) {
        check_success = false;
      }
      if (split_info.left_sum_hessian + split_info.right_sum_hessian != leaf_splits->sum_hessians()) {
        check_success = false;
      }
      if (split_info.left_count + split_info.right_count != leaf_splits->num_data_in_leaf()) {
        check_success = false;
      }
    }
    if (!check_success) {
      if (right_real_leaf_index != -1) {
        const std::unique_ptr<LeafSplits>& leaf_splits = old_level_leaf_splits[i];
        Log::Warning("leaf_splits->sum_gradient() = %f", leaf_splits->sum_gradients());
        Log::Warning("leaf_splits->sum_hessian() = %f", leaf_splits->sum_hessians());
        Log::Warning("leaf_splits->num_data_in_leaf() = %d", leaf_splits->num_data_in_leaf());
        Log::Warning("left_sum_gradient = %f, best_split_info.left_sum_gradient = %f", sum_gradients[left_real_leaf_index], split_info.left_sum_gradient);
        Log::Warning("left_sum_hessian = %f, best_split_info.left_sum_hessian = %f", sum_hessians[left_real_leaf_index], split_info.left_sum_hessian);
        Log::Warning("left_num_data = %d, best_split_info.left_num_data = %d", num_datas[left_real_leaf_index], split_info.left_count);
        Log::Warning("right_sum_gradient = %f, best_split_info.right_sum_gradient = %f", sum_gradients[right_real_leaf_index], split_info.right_sum_gradient);
        Log::Warning("right_sum_hessian = %f, best_split_info.right_sum_hessian = %f", sum_hessians[right_real_leaf_index], split_info.right_sum_hessian);
        Log::Warning("right_num_data = %d, best_split_info.right_num_data = %d", num_datas[right_real_leaf_index], split_info.right_count);
      }
    }
    CHECK(check_success);
  }
  Log::Warning("split test passes with %d leaves", num_leaves_in_cur_level_);
}

bool SymmetricTreeLearner::SplitLevel(Tree* tree) {
  global_timer.Start("symmetric_data_partition_->Split");
  symmetric_data_partition_->Split(
    train_data_,
    best_inner_feature_index_cur_level_,
    best_threshold_cur_level_,
    best_split_default_left_cur_level_,
    best_leaf_in_level_should_be_split_,
    best_level_split_info_);
  global_timer.Stop("symmetric_data_partition_->Split");
  if (best_inner_feature_index_cur_level_ != -1) {
    std::vector<int> old_left_child, old_right_child;
    std::vector<int> old_leaf_indices_in_cur_level = leaf_indices_in_cur_level_;
    std::vector<std::unique_ptr<LeafSplits>> old_level_leaf_splits(num_leaves_in_cur_level_);
    for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
      old_level_leaf_splits[i].reset(level_leaf_splits_[i].release());
    }
    int num_leaves_in_next_level = 0;
    for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level_; ++leaf_index_in_level) {
      const int real_leaf_index = old_leaf_indices_in_cur_level[leaf_index_in_level];
      if (best_leaf_in_level_should_be_split_[leaf_index_in_level]) {
        //TODO(shiyu1994) may be change the type of best_threshold_cur_level_;
        const uint32_t uint32_best_threshold_cur_level = static_cast<uint32_t>(best_threshold_cur_level_);
        SplitInfo& split_info = best_level_split_info_[leaf_index_in_level];
        const int right_leaf_index = tree->Split(real_leaf_index, best_inner_feature_index_cur_level_,
          train_data_->RealFeatureIndex(best_inner_feature_index_cur_level_),
          uint32_best_threshold_cur_level,
          train_data_->RealThreshold(best_inner_feature_index_cur_level_, uint32_best_threshold_cur_level),
          split_info.left_output, split_info.right_output,
          split_info.left_count, split_info.right_count,
          split_info.left_sum_hessian, split_info.right_sum_hessian,
          split_info.gain,
          train_data_->FeatureBinMapper(best_inner_feature_index_cur_level_)->missing_type(),
          split_info.default_left);
        old_left_child.push_back(real_leaf_index);
        old_right_child.push_back(right_leaf_index);
        symmetric_data_partition_->SplitInnerLeafIndex(real_leaf_index, real_leaf_index, right_leaf_index,
          num_leaves_in_next_level, num_leaves_in_next_level + 1);
        const data_size_t left_count = symmetric_data_partition_->leaf_count(real_leaf_index);
        const data_size_t right_count = symmetric_data_partition_->leaf_count(right_leaf_index);
        paired_leaf_indices_in_cur_level_[leaf_index_in_level].resize(2);
        if (split_info.left_count <= split_info.right_count) {
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = real_leaf_index;
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][1] = right_leaf_index;
        } else {
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = right_leaf_index;
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][1] = real_leaf_index;
        }
        // correct leaf count in split info, which was originally estimated from sum of hessians
        split_info.left_count = left_count;
        split_info.right_count = right_count;
        level_leaf_splits_[num_leaves_in_next_level].reset(new LeafSplits(left_count, config_));
        level_leaf_splits_[num_leaves_in_next_level]->Init(
          real_leaf_index,
          split_info.left_count,
          split_info.left_sum_gradient,
          split_info.left_sum_hessian,
          split_info.left_output);
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = real_leaf_index;
        level_leaf_splits_[num_leaves_in_next_level].reset(new LeafSplits(right_count, config_));
        level_leaf_splits_[num_leaves_in_next_level]->Init(
          right_leaf_index,
          split_info.right_count,
          split_info.right_sum_gradient,
          split_info.right_sum_hessian,
          split_info.right_output);
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = right_leaf_index;
      } else {
        old_left_child.push_back(real_leaf_index);
        old_right_child.push_back(-1);
        // update inner leaf index map of data partition
        symmetric_data_partition_->SplitInnerLeafIndex(real_leaf_index, -1, -1, num_leaves_in_next_level, -1);
        paired_leaf_indices_in_cur_level_[leaf_index_in_level].resize(1);
        paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = real_leaf_index;
        level_leaf_splits_[num_leaves_in_next_level].reset(old_level_leaf_splits[leaf_index_in_level].release());
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = real_leaf_index;
      }
    }
    const int num_leaves_in_old_level = num_leaves_in_cur_level_;
    num_pairs_ = num_leaves_in_cur_level_;
    num_leaves_in_cur_level_ = num_leaves_in_next_level;
    //CheckSplit(num_leaves_in_old_level, old_left_child, old_right_child, old_level_leaf_splits);
    return true;
  } else {
    return false;
  }
}

void SymmetricTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
  bool is_constant_hessian,
  bool reset_multi_val_bin) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  CHECK_EQ(num_features_, train_data_->num_features());

  if (reset_multi_val_bin) {
    col_sampler_.SetTrainingData(train_data_);
    GetShareStates(train_data_, is_constant_hessian, false);
  }

  ordered_gradients_.resize(num_data_, 0.0f);
  ordered_hessians_.resize(num_data_, 0.0f);

  //TODO(shiyu1994): handle cost efficient (cegb_)
}

Tree* SymmetricTreeLearner::FitByExistingTree(const Tree* /*old_tree*/,
  const score_t* /*gradients*/, const score_t* /*hessians*/) const {
  //TODO(shiyu1994)
  return nullptr;
}

Tree* SymmetricTreeLearner::FitByExistingTree(const Tree* /*old_tree*/, const std::vector<int>& /*leaf_pred*/,
  const score_t* /*gradients*/, const score_t* /*hessians*/) const {
  // TODO(shiyu1994)
  return nullptr;
}

void SymmetricTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  CHECK_LE(tree->num_leaves(), symmetric_data_partition_->num_leaves());
  if (tree->num_leaves() <= 1) {
    return;
  }
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, tree, out_score] (int /*thread_id*/, data_size_t start, data_size_t end) {
    for (data_size_t i = start; i < end; ++i) {
      const int real_leaf_index = symmetric_data_partition_->GetDataLeafIndex(i);
      const double output = static_cast<double>(tree->LeafOutput(real_leaf_index));
      out_score[i] += output;
    }
  });
}

void SymmetricTreeLearner::RenewTreeOutput(Tree* /*tree*/, const ObjectiveFunction* obj,
  std::function<double(const label_t*, int)> /*residual_getter*/,
  data_size_t /*total_num_data*/, const data_size_t* /*bag_indices*/, data_size_t /*bag_cnt*/) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    Log::Fatal("renew output is not supported with symmetric tree yet");
  }
}

void SymmetricTreeLearner::SetBaggingData(const Dataset* /*subset*/,
  const data_size_t* /*used_indices*/, data_size_t /*num_data*/) {
  Log::Fatal("bagging is not supported with symmetric tree yet");
}

}  // namespace LightGBM
