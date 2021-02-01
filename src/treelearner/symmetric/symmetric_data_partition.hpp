/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_SYMMETRIC_DATA_PARTITION_HPP_
#define LIGHTGBM_TREELEARNER_SYMMETRIC_DATA_PARTITION_HPP_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>

#include "symmetric_feature_histogram.hpp"

namespace LightGBM {

class SymmetricDataPartition {
 public:
  SymmetricDataPartition(data_size_t num_data, int max_num_leaves, int num_threads);

  void ConstructLevelHistograms(std::vector<FeatureHistogram*>* level_feature_histogram,
                                const Dataset* train_data,
                                const std::vector<int8_t>& is_feature_used,
                                const score_t* gradients, const score_t* hessians,
                                score_t* ordered_gradients, score_t* ordered_hessians,
                                TrainingShareStates* share_state) const;

  void Split(const Dataset* train_data,
    const int inner_feature_index,
    const int threshold, const int8_t default_left,
    const std::vector<int8_t>& leaf_should_be_split,
    const std::vector<SplitInfo>& level_split_info);

  void Init();

  void SplitInnerLeafIndex(const int parent_real_leaf_index,
                           const int left_real_leaf_index,
                           const int right_real_leaf_index,
                           const int left_leaf_position,
                           const int right_leaf_position);

  int leaf_count(const int leaf_index) { return leaf_count_[inner_leaf_index_[leaf_index]]; }

  int num_leaves() { return num_leaf_in_level_; }

  int GetDataLeafIndex(const data_size_t data_index) {
    const int inner_leaf_index = data_index_to_leaf_index_[data_index];
    const int real_leaf_index = real_leaf_index_[inner_leaf_index];
    return real_leaf_index;
  }

 private:
  void GetUsedFeatureGroups(const std::vector<int8_t>& is_feature_used,
                            const Dataset* train_data,
                            std::vector<int>* used_feature_groups,
                            int* used_multi_val_feature_group) const;

  template <bool DEFAULT_LEFT, bool MOST_FREQ_LEFT>
  void SplitLevelLeafIndices(
    const Dataset* train_data,
    const int inner_feature_index,
    const uint32_t threshold,
    const std::vector<int8_t>& leaf_should_be_split,
    const std::vector<SplitInfo>& level_split_info,
    const uint32_t most_freq_bin,
    const uint32_t max_bin,
    const uint32_t zero_bin);

  template <bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool DEFAULT_LEFT, bool MOST_FREQ_LEFT>
  void SplitLevelLeafIndicesInner(
    const Dataset* train_data,
    const int inner_feature_index,
    const uint32_t threshold,
    const std::vector<int8_t>& leaf_should_be_split,
    const std::vector<SplitInfo>& level_split_info,
    const uint32_t most_freq_bin,
    const uint32_t max_bin,
    const uint32_t zero_bin);

  std::vector<uint32_t> ordered_small_leaf_index_;
  std::vector<data_size_t> data_indices_in_small_leaf_;
  std::vector<int8_t> is_data_in_small_leaf_;
  data_size_t num_data_in_small_leaf_;
  const data_size_t num_data_;
  bool is_col_wise_;

  std::vector<uint32_t> data_index_to_leaf_index_;
  std::vector<int> leaf_count_;
  std::vector<std::vector<int>> thread_leaf_count_;
  std::vector<int> small_leaf_positions_;
  int num_leaf_in_level_;

  std::vector<int> left_child_index_;
  std::vector<int> right_child_index_;
  std::vector<bool> left_child_smaller_;

  const int num_threads_;
  std::vector<int> thread_data_in_small_leaf_pos_;
  std::vector<int> inner_leaf_index_;
  std::vector<int> position_to_inner_leaf_index_;
  std::vector<int> inner_leaf_index_to_position_;
  std::vector<int> real_leaf_index_;

  int num_small_leaf_;
};

} //  namespace LightGBM

#endif //  LIGHTGBM_TREELEARNER_SYMMETRIC_DATA_PARTITION_HPP_
