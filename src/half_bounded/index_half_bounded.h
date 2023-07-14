/**
 * @file index_half_bounded.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Index for half-bounded range filtering search.
 * Lossless compression on N hnsw on search space
 * @date 2023-06-29
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <ctime>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>

#include "../delta_base_hnsw/hnswalg.h"
#include "../delta_base_hnsw/hnswlib.h"
#include "base_struct.h"
#include "data_wrapper.h"
// #include "hnswalg.h"
// #include "hnswlib.h"
#include "range_index_base.h"
#include "utils.h"

using namespace delta_index_hnsw_full_reverse;
#define INT_MAX __INT_MAX__

namespace halfrangeindex {

template <typename dist_t>
class HalfBoundedHNSW : public HierarchicalNSW<float> {
 public:
  HalfBoundedHNSW(const BaseIndex::IndexParams &index_params,
                  SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16,
                  size_t ef_construction = 200, size_t random_seed = 100)
      : HierarchicalNSW(s, max_elements, M, index_params.ef_construction,
                        random_seed) {
    params = &index_params;
    // in ons-side segment graph, ef_max_ equal to ef_construction
    ef_max_ = index_params.ef_construction;
    ef_basic_construction_ = index_params.ef_construction;
    ef_construction = index_params.ef_construction;
  }

  const BaseIndex::IndexParams *params;
  vector<vector<NeighborLifeCycle<dist_t>>> *range_nns;

  virtual tableint mutuallyConnectNewElementLevel0(
      const void *data_point, tableint cur_c,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      int level, bool isUpdate) {
    size_t Mcurmax = this->maxM0_;
    getNeighborsByHeuristic2(top_candidates, this->M_);
    if (top_candidates.size() > this->M_)
      throw std::runtime_error(
          "Should be not be more than M_ candidates returned by the "
          "heuristic");

    // forward neighbors in top candidates
    int external_id = this->getExternalLabel(cur_c);
    auto nns = &range_nns->at(external_id);

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(this->M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);

      int external_nn = this->getExternalLabel(selectedNeighbors.back());
      NeighborLifeCycle<dist_t> nn(external_nn, top_candidates.top().first,
                                   external_nn, external_nn);
      nns->emplace_back(nn);

      top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    {
      linklistsizeint *ll_cur;
      if (level == 0)
        ll_cur = this->get_linklist0(cur_c);
      else
        ll_cur = this->get_linklist(cur_c, level);

      if (*ll_cur && !isUpdate) {
        throw std::runtime_error(
            "The newly inserted element should have blank link list");
      }
      this->setListCount(ll_cur, selectedNeighbors.size());
      tableint *data = (tableint *)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        if (data[idx] && !isUpdate)
          throw std::runtime_error("Possible memory corruption");
        if (level > this->element_levels_[selectedNeighbors[idx]])
          throw std::runtime_error(
              "Trying to make a link on a non-existent level");

        data[idx] = selectedNeighbors[idx];
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      std::unique_lock<std::mutex> lock(
          this->link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint *ll_other;
      if (level == 0)
        ll_other = this->get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = this->get_linklist(selectedNeighbors[idx], level);

      size_t sz_link_list_other = this->getListCount(ll_other);

      if (sz_link_list_other > Mcurmax)
        throw std::runtime_error("Bad value of sz_link_list_other");
      if (selectedNeighbors[idx] == cur_c)
        throw std::runtime_error("Trying to connect an element to itself");
      if (level > this->element_levels_[selectedNeighbors[idx]])
        throw std::runtime_error(
            "Trying to make a link on a non-existent level");

      tableint *data = (tableint *)(ll_other + 1);

      bool is_cur_c_present = false;
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (data[j] == cur_c) {
            is_cur_c_present = true;
            break;
          }
        }
      }

      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or
      // run the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          this->setListCount(ll_other, sz_link_list_other + 1);

          // add external_id to nns (backward nn)
          int external_backward_id =
              this->getExternalLabel(selectedNeighbors[idx]);
          NeighborLifeCycle<dist_t> backward_nn(external_id, 0, external_id,
                                                external_id);
          range_nns->at(external_backward_id).emplace_back(backward_nn);

        } else {
          // finding the "weakest" element to replace it with the new one
          dist_t d_max =
              fstdistfunc_(this->getDataByInternalId(cur_c),
                           this->getDataByInternalId(selectedNeighbors[idx]),
                           this->dist_func_param_);
          // Heuristic:
          std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
              candidates;
          candidates.emplace(d_max, cur_c);

          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(
                fstdistfunc_(this->getDataByInternalId(data[j]),
                             this->getDataByInternalId(selectedNeighbors[idx]),
                             this->dist_func_param_),
                data[j]);
          }

          getNeighborsByHeuristic2(candidates, Mcurmax);

          // Backward nns in candidates
          {
            // cout << "id: " << getExternalLabel(selectedNeighbors[idx]) <<
            // endl;
            auto back_nns =
                &range_nns->at(this->getExternalLabel(selectedNeighbors[idx]));
            auto temp_candidates(candidates);

            std::vector<int> survive_nns;  // nns kept after pruning

            while (!temp_candidates.empty()) {
              auto temp_id = temp_candidates.top().second;
              survive_nns.emplace_back(getExternalLabel(temp_id));
              temp_candidates.pop();
            }

            for (unsigned ii = 0; ii < back_nns->size(); ii++) {
              auto nn = &back_nns->at(ii);
              if (nn->start == nn->end) {
                if (std::find(survive_nns.begin(), survive_nns.end(), nn->id) ==
                    survive_nns.end()) {
                  // nn not survive
                  nn->end = external_id;
                }
              }
            }

            // cur_c could be inserted
            if (std::find(survive_nns.begin(), survive_nns.end(),
                          external_id) != survive_nns.end()) {
              // add external_id to nns (backward nn)
              NeighborLifeCycle<dist_t> backward_nn(external_id, 0, external_id,
                                                    external_id);
              back_nns->emplace_back(backward_nn);
            }
          }

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }

          this->setListCount(ll_other, indx);
          // Nearest K:
          /*int indx = -1;
          for (int j = 0; j < sz_link_list_other; j++) {
              dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
          getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
                  indx = j;
                  d_max = d;
              }
          }
          if (indx >= 0) {
              data[indx] = cur_c;
          } */
        }
      }
    }

    return next_closest_entry_point;
  }
};

template <typename dist_t>
class HalfBoundedIndex : public BaseIndex {
 public:
  vector<vector<NeighborLifeCycle<dist_t>>> indexed_arr;

  HalfBoundedIndex(delta_index_hnsw_full_reverse::SpaceInterface<dist_t> *s,
                   const DataWrapper *data)
      : BaseIndex(data) {
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    index_info = new IndexInfo();
    index_info->index_version_type = "HalfBoundedIndex";
  }

  delta_index_hnsw_full_reverse::DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_;

  VisitedListPool *visited_list_pool_;
  IndexInfo *index_info;

  void printOnebatch(int pos = -1) {
    if (pos == -1) {
      pos = data_wrapper->data_size / 2;
    }

    cout << "nns at position: " << pos << endl;
    for (auto nns : indexed_arr[pos]) {
      cout << nns.id << "(" << nns.start << "," << nns.end << ")\n" << endl;
    }
    cout << endl;
  }

  void countNeighbrs() {
    double batch_counter = 0;
    double max_batch_counter = 0;
    if (!indexed_arr.empty())
      for (unsigned j = 0; j < indexed_arr.size(); j++) {
        int temp_size = 0;
        temp_size += indexed_arr[j].size();
        index_info->nodes_amount += temp_size;
      }
    index_info->avg_forward_nns =
        index_info->nodes_amount / (float)data_wrapper->data_size;
    if (isLog) {
      cout << "Avg. nn #: "
           << index_info->nodes_amount / (float)data_wrapper->data_size << endl;
      batch_counter = 0;
    }

    index_info->avg_reverse_nns = 0;
  }

  void buildIndex(const IndexParams *index_params) override {
    cout << "Building Index using " << index_info->index_version_type << endl;
    timeval tt1, tt2;
    visited_list_pool_ = new delta_index_hnsw_full_reverse::VisitedListPool(
        1, data_wrapper->data_size);

    // build HNSW
    L2Space space(data_wrapper->data_dim);
    HalfBoundedHNSW<float> hnsw(
        *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
        index_params->ef_construction, index_params->random_seed);

    indexed_arr.clear();
    indexed_arr.resize(data_wrapper->data_size);

    hnsw.range_nns = &indexed_arr;

    gettimeofday(&tt1, NULL);
    // #pragma omp parallel for
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      hnsw.addPoint(data_wrapper->nodes.at(i).data(), i);
    }
    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);

    // count neighbors number
    countNeighbrs();

    if (index_params->print_one_batch) {
      printOnebatch();
    }
  }

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

      // search cw on the fly
      // vector<int> current_neighbors;
      // current_neighbors = decompressDeltaPath(
      //     indexed_arr[current_node_id], 0, query_bound.second,
      //     current_node_id);

      auto neighbor_it = indexed_arr.at(current_node_id).begin();

      gettimeofday(&tt1, NULL);

      while (neighbor_it != indexed_arr[current_node_id].end()) {
        if ((neighbor_it->id < query_bound.second) &&
            (neighbor_it->end == neighbor_it->start ||
             neighbor_it->end >= query_bound.second)) {
          int candidate_id = neighbor_it->id;

          if (!(visited_array[candidate_id] == visited_array_tag)) {
            visited_array[candidate_id] = visited_array_tag;
            float dist = fstdistfunc_(query.data(),
                                      data_wrapper->nodes[candidate_id].data(),
                                      dist_func_param_);

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
        neighbor_it++;
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
  // This is bad for half bounded search.
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

      auto neighbor_it = indexed_arr.at(current_node_id).begin();
      gettimeofday(&tt1, NULL);

      while (neighbor_it != indexed_arr[current_node_id].end()) {
        if ((neighbor_it->id < query_bound.second)) {
          int candidate_id = neighbor_it->id;

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
        neighbor_it++;
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

  ~HalfBoundedIndex() {
    delete index_info;
    indexed_arr.clear();
    delete visited_list_pool_;
  }
};

}  // namespace halfrangeindex