/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TRAIN_SHARE_STATES_H_
#define LIGHTGBM_TRAIN_SHARE_STATES_H_

#include <LightGBM/bin.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace LightGBM {

class MultiValBinWrapper {
 public:
  MultiValBinWrapper(MultiValBin* bin, data_size_t num_data,
    const std::vector<int>& feature_groups_contained);

  bool IsSparse() {
    if (multi_val_bin_ != nullptr) {
      return multi_val_bin_->IsSparse();
    }
    return false;
  }

  void InitTrain(const std::vector<int>& group_feature_start,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* bagging_use_indices,
    data_size_t bagging_indices_cnt);

  void HistMove(const std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>& hist_buf,
                hist_t* origin_hist_data);

  void HistMerge(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
                 hist_t* origin_hist_data);

  void ResizeHistBuf(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
    MultiValBin* sub_multi_val_bin);

  void SymmetricTreeResizeHistBuf(std::vector<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>>* hist_buf,
    MultiValBin* sub_multi_val_bin,
    const std::vector<hist_t*>& origin_hist_data,
    const int num_leaf);

  template <bool USE_INDICES, bool ORDERED>
  void ConstructHistograms(const data_size_t* data_indices,
      data_size_t num_data,
      const score_t* gradients,
      const score_t* hessians,
      std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
      hist_t* origin_hist_data) {
    const auto cur_multi_val_bin = (is_use_subcol_ || is_use_subrow_)
          ? multi_val_bin_subset_.get()
          : multi_val_bin_.get();
    if (cur_multi_val_bin != nullptr) {
      global_timer.Start("Dataset::sparse_bin_histogram");
      n_data_block_ = 1;
      data_block_size_ = num_data;
      Threading::BlockInfo<data_size_t>(num_threads_, num_data, min_block_size_,
                                        &n_data_block_, &data_block_size_);
      ResizeHistBuf(hist_buf, cur_multi_val_bin);
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) num_threads(num_threads_)
      for (int block_id = 0; block_id < n_data_block_; ++block_id) {
        OMP_LOOP_EX_BEGIN();
        data_size_t start = block_id * data_block_size_;
        data_size_t end = std::min<data_size_t>(start + data_block_size_, num_data);
        ConstructHistogramsForBlock<USE_INDICES, ORDERED>(
          cur_multi_val_bin, start, end, data_indices, gradients, hessians,
          block_id, hist_buf, origin_hist_data);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      global_timer.Stop("Dataset::sparse_bin_histogram");

      global_timer.Start("Dataset::sparse_bin_histogram_merge");
      HistMerge(hist_buf, origin_hist_data);
      global_timer.Stop("Dataset::sparse_bin_histogram_merge");
      global_timer.Start("Dataset::sparse_bin_histogram_move");
      HistMove(*hist_buf, origin_hist_data);
      global_timer.Stop("Dataset::sparse_bin_histogram_move");
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructSymmetricTreeHistograms(
      const data_size_t* data_indices,
      const uint32_t* leaf_indices,
      data_size_t num_data,
      const score_t* gradients,
      const score_t* hessians,
      std::vector<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>>* hist_buf,
      const std::vector<hist_t*>& origin_hist_data) {
    const auto cur_multi_val_bin = (is_use_subcol_ || is_use_subrow_)
          ? multi_val_bin_subset_.get()
          : multi_val_bin_.get();
    if (cur_multi_val_bin != nullptr) {
      global_timer.Start("Dataset::sparse_bin_histogram");
      n_data_block_ = 1;
      data_block_size_ = num_data;
      Threading::BlockInfo<data_size_t>(num_threads_, num_data, min_block_size_,
                                        &n_data_block_, &data_block_size_);
      hist_buf->resize(origin_hist_data.size());
      for (size_t i = 0; i < origin_hist_data.size(); ++i) {
        ResizeHistBuf(&((*hist_buf)[i]), cur_multi_val_bin);
      }
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) num_threads(num_threads_)
      for (int block_id = 0; block_id < n_data_block_; ++block_id) {
        OMP_LOOP_EX_BEGIN();
        data_size_t start = block_id * data_block_size_;
        data_size_t end = std::min<data_size_t>(start + data_block_size_, num_data);
        ConstructSymmetricTreeHistogramsForBlock<USE_INDICES, ORDERED>(
          cur_multi_val_bin, start, end, data_indices, leaf_indices, gradients, hessians,
          block_id, hist_buf, origin_hist_data);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      global_timer.Stop("Dataset::sparse_bin_histogram");

      global_timer.Start("Dataset::sparse_bin_histogram_merge");
      for (size_t i = 0; i < origin_hist_data.size(); ++i) {
        HistMerge(&((*hist_buf)[i]), origin_hist_data[i]);
      }
      global_timer.Stop("Dataset::sparse_bin_histogram_merge");
      global_timer.Start("Dataset::sparse_bin_histogram_move");
      for (size_t i = 0; i < origin_hist_data.size(); ++i) {
        HistMove((*hist_buf)[i], origin_hist_data[i]);
      }
      global_timer.Stop("Dataset::sparse_bin_histogram_move");
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
    hist_t* origin_hist_data) {
    hist_t* data_ptr = origin_hist_data;
    if (block_id == 0) {
      if (is_use_subcol_) {
        data_ptr = hist_buf->data() + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
      }
    } else {
      data_ptr = hist_buf->data() +
        static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
    }
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHistBufferEntrySize);
    if (USE_INDICES) {
      if (ORDERED) {
        sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructHistogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructSymmetricTreeHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices, const uint32_t* leaf_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    std::vector<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>>* hist_buf,
    const std::vector<hist_t*>& origin_hist_data) {
    std::vector<hist_t*> data_ptr(origin_hist_data.size(), nullptr);
    for (size_t i = 0; i < origin_hist_data.size(); ++i) {
      data_ptr[i] = origin_hist_data[i];
    }
    auto& hist_buf_ref = *hist_buf;
    for (size_t i = 0; i < origin_hist_data.size(); ++i) {
      if (block_id == 0) {
        if (is_use_subcol_) {
          data_ptr[i] = hist_buf_ref[i].data() + hist_buf_ref[i].size() - 2 * static_cast<size_t>(num_bin_aligned_);
        }
      } else {
        data_ptr[i] = hist_buf_ref[i].data() +
          static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
      }
    }
    for (size_t i = 0; i < data_ptr.size(); ++i) {
      std::memset(reinterpret_cast<void*>(data_ptr[i]), 0, num_bin_ * kHistBufferEntrySize);
    }
    if (USE_INDICES) {
      if (ORDERED) {
        sub_multi_val_bin->ConstructSymmetricTreeHistogramOrdered(start, end, data_indices,
                                                gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructSymmetricTreeHistogram(start, end, data_indices, leaf_indices, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructSymmetricTreeHistogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  void CopyMultiValBinSubset(const std::vector<int>& group_feature_start,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* bagging_use_indices,
    data_size_t bagging_indices_cnt);

  void SetUseSubrow(bool is_use_subrow) {
    is_use_subrow_ = is_use_subrow;
  }

  void SetSubrowCopied(bool is_subrow_copied) {
    is_subrow_copied_ = is_subrow_copied;
  }

 private:
  bool is_use_subcol_ = false;
  bool is_use_subrow_ = false;
  bool is_subrow_copied_ = false;
  std::unique_ptr<MultiValBin> multi_val_bin_;
  std::unique_ptr<MultiValBin> multi_val_bin_subset_;
  std::vector<uint32_t> hist_move_src_;
  std::vector<uint32_t> hist_move_dest_;
  std::vector<uint32_t> hist_move_size_;
  const std::vector<int> feature_groups_contained_;

  int num_threads_;
  int num_bin_;
  int num_bin_aligned_;
  int n_data_block_;
  int data_block_size_;
  int min_block_size_;
  int num_data_;

  const size_t kHistBufferEntrySize = 2 * sizeof(hist_t);
};

struct TrainingShareStates {
  int num_threads = 0;
  bool is_col_wise = true;
  bool is_constant_hessian = true;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;

  TrainingShareStates() {
    multi_val_bin_wrapper_.reset(nullptr);
  }

  int num_hist_total_bin() { return num_hist_total_bin_; }

  const std::vector<uint32_t>& feature_hist_offsets() { return feature_hist_offsets_; }

  bool IsSparseRowwise() {
    return (multi_val_bin_wrapper_ != nullptr && multi_val_bin_wrapper_->IsSparse());
  }

  void SetMultiValBin(MultiValBin* bin, data_size_t num_data,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    bool dense_only, bool sparse_only);

  void CalcBinOffsets(const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    std::vector<uint32_t>* offsets, bool is_col_wise);

  void InitTrain(const std::vector<int>& group_feature_start,
        const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
        const std::vector<int8_t>& is_feature_used) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->InitTrain(group_feature_start,
        feature_groups,
        is_feature_used,
        bagging_use_indices,
        bagging_indices_cnt);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructHistograms(const data_size_t* data_indices,
                          data_size_t num_data,
                          const score_t* gradients,
                          const score_t* hessians,
                          hist_t* hist_data) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->ConstructHistograms<USE_INDICES, ORDERED>(
        data_indices, num_data, gradients, hessians, &hist_buf_, hist_data);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructSymmetricTreeHistograms(const data_size_t* data_indices,
                          const uint32_t* leaf_indices,
                          data_size_t num_small_leaf_data,
                          const score_t* gradients,
                          const score_t* hessians,
                          const std::vector<hist_t*>& hist_data) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->ConstructSymmetricTreeHistograms<USE_INDICES, ORDERED>(
        data_indices, leaf_indices, num_small_leaf_data, gradients, hessians, &symmetric_tree_hist_buf_, hist_data);
    }
  }

  void SetUseSubrow(bool is_use_subrow) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->SetUseSubrow(is_use_subrow);
    }
  }

  void SetSubrowCopied(bool is_subrow_copied) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->SetSubrowCopied(is_subrow_copied);
    }
  }

 private:
  std::vector<uint32_t> feature_hist_offsets_;
  int num_hist_total_bin_ = 0;
  std::unique_ptr<MultiValBinWrapper> multi_val_bin_wrapper_;
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> hist_buf_;
  std::vector<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>> symmetric_tree_hist_buf_;
  int num_total_bin_ = 0;
  double num_elements_per_row_ = 0.0f;
};

}  // namespace LightGBM

#endif   // LightGBM_TRAIN_SHARE_STATES_H_
