/**
 * @file index_recursion_batch.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Index for arbitrary range filtering search
 * Search more neighbors and insert them into the graph
 * Prune the reverse nns and also use batch to store them
 * @date 2023-06-20
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <ctime>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>

#include "base_struct.h"
#include "data_wrapper.h"
#include "delta_base_hnsw/hnswalg.h"
#include "delta_base_hnsw/hnswlib.h"
#include "index_recursion_batch.h"
#include "range_index_base.h"
#include "utils.h"

using namespace delta_index_hnsw_full_reverse;
#define INT_MAX __INT_MAX__

namespace rangeindex {

class RecursionIndexPruneReverse : public RecursionIndex {
 public:
  RecursionIndexPruneReverse(
      delta_index_hnsw_full_reverse::SpaceInterface<float> *s,
      const DataWrapper *data)
      : RecursionIndex(s, data) {
    index_info->index_version_type = "RecursionIndexPruneReverse";
  }

  // do pruning all sth else. In this base version, just no prune and collect
  // all reverse neighbor in one batch.
  void processReverseNeighbors(RangeFilteringHNSW<float> &hnsw,
                               const unsigned prune_K) {
    // Pruning reverse nns and add them to index
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      auto before_pruning_candidates = hnsw.reverse_nns_vecs.at(i);
      // auto after_pruning_candidates = hnsw.reverse_nns_vecs[i];
      size_t batch_counter = 0;
      while (before_pruning_candidates.size() > 0) {
        OneBatchNeighbors batchNN(batch_counter);
        vector<pair<int, float>> after_pruning_candidates =
            before_pruning_candidates;
        heuristicPrune(data_wrapper->nodes, after_pruning_candidates, prune_K);
        vector<pair<int, float>> leftover_candidates;

        auto external_left_most =
            std::min_element(
                after_pruning_candidates.cbegin(),
                after_pruning_candidates.cend(),
                [](const pair<int, float> &lhs, const pair<int, float> &rhs) {
                  return lhs.first < rhs.first;
                })
                ->first;

        for (auto nn : before_pruning_candidates) {
          if (nn.first < external_left_most) {
            leftover_candidates.emplace_back(nn);
          }
        }
        before_pruning_candidates = leftover_candidates;

        batchNN.nns.assign(after_pruning_candidates.begin(),
                           after_pruning_candidates.end());

        for (auto nn : batchNN.nns) {
          batchNN.nns_id.emplace_back(nn.first);
        }
        batchNN.start = external_left_most;
        if (batch_counter) {
          batchNN.end = directed_indexed_arr.at(i).reverse_nns.back().start;
        } else {
          batchNN.end = data_wrapper->data_size;
        }
        directed_indexed_arr.at(i).reverse_nns.emplace_back(batchNN);

        batch_counter++;
      }
    }
  }

  void countNeighbrs() {
    double batch_counter = 0;
    double max_batch_counter = 0;
    if (!directed_indexed_arr.empty())
      for (unsigned j = 0; j < directed_indexed_arr.size(); j++) {
        int temp_size = 0;
        for (auto nns : directed_indexed_arr[j].forward_nns) {
          temp_size += nns.nns_id.size();
        }
        batch_counter += directed_indexed_arr[j].forward_nns.size();
        index_info->nodes_amount += temp_size;
      }
    index_info->avg_forward_nns =
        index_info->nodes_amount / (float)data_wrapper->data_size;
    if (isLog) {
      cout << "Max. forward batch nn #: " << max_batch_counter << endl;
      cout << "Avg. forward nn #: "
           << index_info->nodes_amount / (float)data_wrapper->data_size << endl;
      cout << "Avg. forward batch #: "
           << batch_counter / (float)data_wrapper->data_size << endl;
      batch_counter = 0;
    }

    int reverse_node_amount = 0;
    if (!directed_indexed_arr.empty()) {
      for (unsigned j = 0; j < directed_indexed_arr.size(); j++) {
        int temp_size = 0;
        for (auto nns : directed_indexed_arr[j].reverse_nns) {
          temp_size += nns.nns_id.size();
        }
        reverse_node_amount += temp_size;
        batch_counter += directed_indexed_arr[j].reverse_nns.size();
      }
    }

    index_info->nodes_amount += reverse_node_amount;
    index_info->avg_reverse_nns =
        reverse_node_amount / (float)data_wrapper->data_size;
    if (isLog) {
      cout << "Avg. reverse nn #: "
           << reverse_node_amount / (float)data_wrapper->data_size << endl;
      cout << "Avg. reverse batch #: "
           << batch_counter / (float)data_wrapper->data_size << endl;
      cout << "Avg. delta nn #: "
           << index_info->nodes_amount / (float)data_wrapper->data_size << endl;
    }
  }

  void buildIndex(const IndexParams *index_params) override {
    cout << "Building Index using " << index_info->index_version_type << endl;
    timeval tt1, tt2;
    visited_list_pool_ = new delta_index_hnsw_full_reverse::VisitedListPool(
        1, data_wrapper->data_size);

    // build HNSW
    L2Space space(data_wrapper->data_dim);
    RangeFilteringHNSW<float> hnsw(
        *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
        index_params->ef_construction, index_params->random_seed);

    directed_indexed_arr.clear();
    directed_indexed_arr.resize(data_wrapper->data_size);
    hnsw.directed_nns = &directed_indexed_arr;

    hnsw.reverse_nns_vecs.resize(data_wrapper->data_size);
    gettimeofday(&tt1, NULL);
    // #pragma omp parallel for
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      hnsw.addPoint(data_wrapper->nodes.at(i).data(), i);
    }
    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);

    // processing reverse neighbors
    processReverseNeighbors(hnsw, index_params->K);

    // count neighbors number
    countNeighbrs();

    if (index_params->print_one_batch) {
      printOnebatch();
    }
  };

  // range filtering search, only calculate distance on on-range nodes.
  vector<int> rangeFilteringSearchInRange(
      const SearchParams *search_params, SearchInfo *search_info,
      const vector<float> &query,
      const std::pair<int, int> query_bound) override {
    timeval tt1, tt2, tt3, tt4;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    float lower_bound = INT_MAX;
    std::priority_queue<pair<float, int>> top_candidates;
    std::priority_queue<pair<float, int>> candidate_set;

    const int data_size = data_wrapper->data_size;
    const int two_batch_threshold =
        data_size * search_params->control_batch_threshold;
    search_info->total_comparison = 0;
    search_info->internal_search_time = 0;
    search_info->cal_dist_time = 0;
    search_info->fetch_nns_time = 0;
    // finding enters
    vector<int> enter_list;
    {
      int lbound = query_bound.first;
      int interval = (query_bound.second - lbound) / 3;
      for (size_t i = 0; i < 3; i++) {
        int point = lbound + interval * i;
        float dist = EuclideanDistance(data_wrapper->nodes[point], query);
        candidate_set.push(make_pair(-dist, point));
        enter_list.emplace_back(point);
        visited_array[point] = visited_array_tag;
      }
    }
    gettimeofday(&tt3, NULL);

    // 只有一个enter
    // float dist_enter = EuclideanDistance(data_nodes[l_bound], query);
    // candidate_set.push(make_pair(-dist_enter, l_bound));
    // TODO: How to find proper enters.

    size_t hop_counter = 0;

    while (!candidate_set.empty()) {
      std::pair<float, int> current_node_pair = candidate_set.top();
      int current_node_id = current_node_pair.second;

      if (-current_node_pair.first > lower_bound) {
        break;
      }

#ifdef LOG_DEBUG_MODE
      cout << "current node: " << current_node_pair.second << "  -- "
           << -current_node_pair.first << endl;
#endif

      // if (search_info->is_investigate) {
      //   search_info->SavePathInvestigate(current_node_pair.second,
      //                                    -current_node_pair.first,
      //                                    hop_counter, num_search_comparison);
      // }
      hop_counter++;

      candidate_set.pop();

      // // only search when candidate point is inside the range
      // if (current_node_id < l_bound || current_node_id > r_bound) {
      //   // cout << "no satisfied range point" << endl;
      //   continue;
      // }

      // search cw on the fly
      vector<int> current_neighbors;
      vector<vector<OneBatchNeighbors>::const_iterator> neighbor_iterators;

      gettimeofday(&tt1, NULL);
      // current_neighbors = decompressDeltaPath(
      //     directed_indexed_arr[current_node_id].forward_nns,
      //     directed_indexed_arr[current_node_id].reverse_nns, l_bound,
      //     r_bound);
      {
        auto forward_it = decompressForwardPath(
            directed_indexed_arr[current_node_id].forward_nns,
            query_bound.first);
        if (forward_it !=
            directed_indexed_arr[current_node_id].forward_nns.end()) {
          neighbor_iterators.emplace_back(forward_it);
          if (current_node_id - query_bound.first < two_batch_threshold) {
            forward_it++;
            if (forward_it !=
                directed_indexed_arr[current_node_id].forward_nns.end()) {
              neighbor_iterators.emplace_back(forward_it);
            }
          }
        }

        auto reverse_it = decompressReversePath(
            directed_indexed_arr[current_node_id].reverse_nns,
            query_bound.second);
        if (reverse_it !=
            directed_indexed_arr[current_node_id].reverse_nns.end()) {
          neighbor_iterators.emplace_back(reverse_it);
          if (query_bound.second - current_node_id < two_batch_threshold) {
            reverse_it++;
            if (reverse_it !=
                directed_indexed_arr[current_node_id].reverse_nns.end()) {
              neighbor_iterators.emplace_back(reverse_it);
            }
          }
        }
      }

      gettimeofday(&tt2, NULL);
      AccumulateTime(tt1, tt2, search_info->fetch_nns_time);
      // print_set(current_neighbors);
      // assert(false);
      gettimeofday(&tt1, NULL);

      for (auto batch_it : neighbor_iterators) {
        for (auto candidate_id : batch_it->nns_id) {
          if (candidate_id < query_bound.first ||
              candidate_id > query_bound.second)
            continue;
          if (!(visited_array[candidate_id] == visited_array_tag)) {
            visited_array[candidate_id] = visited_array_tag;

            // float dist = EuclideanDistance(query, data_nodes[candidate_id]);
            float dist = fstdistfunc_(query.data(),
                                      data_wrapper->nodes[candidate_id].data(),
                                      dist_func_param_);

#ifdef LOG_DEBUG_MODE
            // cout << "candidate: " << candidate_id << "  dist: " << dist <<
            // endl;
#endif

            num_search_comparison++;
            if (top_candidates.size() < search_params->search_ef ||
                lower_bound > dist) {
              candidate_set.push(make_pair(-dist, candidate_id));
              top_candidates.push(make_pair(dist, candidate_id));
              if (top_candidates.size() > search_params->search_ef) {
                top_candidates.pop();
              }
              if (!top_candidates.empty()) {
                lower_bound = top_candidates.top().first;
              }
            }
          }
        }
      }
      gettimeofday(&tt2, NULL);
      AccumulateTime(tt1, tt2, search_info->cal_dist_time);
    }

    vector<int> res;
    while (top_candidates.size() > search_params->query_K) {
      top_candidates.pop();
    }

    while (!top_candidates.empty()) {
      res.emplace_back(top_candidates.top().second);
      top_candidates.pop();
    }
    search_info->total_comparison += num_search_comparison;

#ifdef LOG_DEBUG_MODE
    print_set(res);
    cout << l_bound << "," << r_bound << endl;
    assert(false);
#endif
    visited_list_pool_->releaseVisitedList(vl);

    gettimeofday(&tt4, NULL);
    CountTime(tt3, tt4, search_info->internal_search_time);

    return res;
  }

  // also calculate outbount dists, similar to knn-first
  vector<int> rangeFilteringSearchOutBound(
      const SearchParams *search_params, SearchInfo *search_info,
      const vector<float> &query,
      const std::pair<int, int> query_bound) override {
    timeval tt1, tt2, tt3, tt4;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    float lower_bound = INT_MAX;
    std::priority_queue<pair<float, int>> top_candidates;
    std::priority_queue<pair<float, int>> candidate_set;

    const int data_size = data_wrapper->data_size;
    search_info->total_comparison = 0;
    search_info->internal_search_time = 0;
    search_info->cal_dist_time = 0;
    search_info->fetch_nns_time = 0;
    num_search_comparison = 0;
    // finding enters
    vector<int> enter_list;
    {
      int lbound = query_bound.first;
      int interval = (query_bound.second - lbound) / 3;
      for (size_t i = 0; i < 3; i++) {
        int point = lbound + interval * i;
        float dist = EuclideanDistance(data_wrapper->nodes[point], query);
        candidate_set.push(make_pair(-dist, point));
        enter_list.emplace_back(point);
        visited_array[point] = visited_array_tag;
      }
    }
    gettimeofday(&tt3, NULL);

    size_t hop_counter = 0;

    while (!candidate_set.empty()) {
      std::pair<float, int> current_node_pair = candidate_set.top();
      int current_node_id = current_node_pair.second;

      if (-current_node_pair.first > lower_bound) {
        break;
      }
      // if (search_info->is_investigate) {
      //   search_info->SavePathInvestigate(current_node_pair.second,
      //                                    -current_node_pair.first,
      //                                    hop_counter, num_search_comparison);
      // }
      hop_counter++;

      candidate_set.pop();
      vector<int> current_neighbors;
      vector<vector<OneBatchNeighbors>::const_iterator> neighbor_iterators;

      gettimeofday(&tt1, NULL);
      {
        auto forward_it = decompressForwardPath(
            directed_indexed_arr[current_node_id].forward_nns,
            query_bound.first);
        if (forward_it !=
            directed_indexed_arr[current_node_id].forward_nns.end()) {
          neighbor_iterators.emplace_back(forward_it);
          // only add one batch in this method
        }

        auto reverse_it = decompressReversePath(
            directed_indexed_arr[current_node_id].reverse_nns,
            query_bound.second);
        if (reverse_it !=
            directed_indexed_arr[current_node_id].reverse_nns.end()) {
          neighbor_iterators.emplace_back(reverse_it);
        }
      }

      gettimeofday(&tt2, NULL);
      AccumulateTime(tt1, tt2, search_info->fetch_nns_time);
      gettimeofday(&tt1, NULL);

      for (auto batch_it : neighbor_iterators) {
        for (auto candidate_id : batch_it->nns_id) {
          if (candidate_id < query_bound.first ||
              candidate_id > query_bound.second)
            continue;
          if (!(visited_array[candidate_id] == visited_array_tag)) {
            visited_array[candidate_id] = visited_array_tag;

            float dist = fstdistfunc_(query.data(),
                                      data_wrapper->nodes[candidate_id].data(),
                                      dist_func_param_);

            num_search_comparison++;
            if (top_candidates.size() < search_params->search_ef ||
                lower_bound > dist) {
              candidate_set.emplace(-dist, candidate_id);
              // add to top_candidates only in range
              if (candidate_id <= query_bound.second &&
                  candidate_id >= query_bound.first) {
                top_candidates.emplace(dist, candidate_id);
                if (top_candidates.size() > search_params->search_ef) {
                  top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                  lower_bound = top_candidates.top().first;
                }
              }
            }
          }
        }
      }
      gettimeofday(&tt2, NULL);
      AccumulateTime(tt1, tt2, search_info->cal_dist_time);
    }

    vector<int> res;
    while (top_candidates.size() > search_params->query_K) {
      top_candidates.pop();
    }

    while (!top_candidates.empty()) {
      res.emplace_back(top_candidates.top().second);
      top_candidates.pop();
    }
    search_info->total_comparison += num_search_comparison;

#ifdef LOG_DEBUG_MODE
    print_set(res);
    cout << l_bound << "," << r_bound << endl;
    assert(false);
#endif
    visited_list_pool_->releaseVisitedList(vl);

    gettimeofday(&tt4, NULL);
    CountTime(tt3, tt4, search_info->internal_search_time);

    return res;
  }

  ~RecursionIndexPruneReverse() {
    // delete index_info;
    // directed_indexed_arr.clear();
    // delete visited_list_pool_;
  }
};
}  // namespace rangeindex