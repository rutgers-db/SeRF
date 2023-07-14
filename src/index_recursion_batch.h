/**
 * @file index_recursion_batch.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Index for arbitrary range filtering search
 * Search more neighbors and insert them into the graph
 * @date 2023-06-19
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
#include "range_index_base.h"
#include "utils.h"

using namespace delta_index_hnsw_full_reverse;
#define INT_MAX __INT_MAX__

namespace rangeindex {

template <typename dist_t>
class RangeFilteringHNSW : public HierarchicalNSW<float> {
 public:
  RangeFilteringHNSW(SpaceInterface<float> *s, size_t max_elements,
                     size_t M = 16, size_t ef_construction = 200,
                     size_t random_seed = 100)
      : HierarchicalNSW(s, max_elements, M, ef_construction, random_seed){};
  RangeFilteringHNSW(const BaseIndex::IndexParams &index_params,
                     SpaceInterface<float> *s, size_t max_elements,
                     size_t M = 16, size_t ef_construction = 200,
                     size_t random_seed = 100)
      : HierarchicalNSW(s, max_elements, M, index_params.ef_construction,
                        random_seed) {
    params = &index_params;
    ef_max_ = index_params.ef_max;
  }

  const BaseIndex::IndexParams *params;
  vector<vector<pair<int, float>>> reverse_nns_vecs;

  // rewrite heuristic pruning for supporting ef_for_pruning
  void getNeighborsByHeuristic2LimitSize(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }
    // top_candidates should contain ef_max_ nns.

    // pop out far nodes
    while (top_candidates.size() >
           params->ef_large_for_pruning) {  // or ef_basic_construction?
      top_candidates.pop();
    }
    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first,
                            top_candidates.top().second);
      top_candidates.pop();
      // if (queue_closest.size() >
      //     ef_for_pruning_) {  // or ef_basic_construction?
      //   break;
      // }
    }
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        empty_q;
    top_candidates.swap(empty_q);

    while (queue_closest.size()) {
      if (return_list.size() >= M) break;
      std::pair<dist_t, tableint> curent_pair = queue_closest.top();
      dist_t dist_to_query = -curent_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, tableint> second_pair : return_list) {
        dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                      getDataByInternalId(curent_pair.second),
                                      dist_func_param_);
        ;
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
  }

  // rewrite connect new element
  virtual tableint mutuallyConnectNewElementLevel0(
      const void *data_point, tableint cur_c,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      int level, bool isUpdate) {
    // recursive add
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        before_prune_cancidates(top_candidates);
    std::unordered_set<tableint> selectedNeighbors_set;
    selectedNeighbors_set.reserve(M_);

    size_t Mcurmax = level ? maxM_ : maxM0_;

    getNeighborsByHeuristic2LimitSize(top_candidates, M_);
    if (top_candidates.size() > M_)
      throw std::runtime_error(
          "Should be not be more than M_ candidates returned by the "
          "heuristic");

    // forward neighbors in top candidates
    int external_id = getExternalLabel(cur_c);
    auto nns = &directed_nns->at(external_id);

    // lifecycle
    OneBatchNeighbors batchNN(0);

    // record left most position;
    int external_left_most = -1;
    vector<int> addedNeighborPositions;

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);

      selectedNeighbors_set.emplace(top_candidates.top().second);

      int external_nn = getExternalLabel(selectedNeighbors.back());

      batchNN.nns.emplace_back(
          make_pair(external_nn, top_candidates.top().first));
      // nns->forward_nns.emplace_back(
      //     make_pair(external_nn, top_candidates.top().first));

      top_candidates.pop();

      addedNeighborPositions.emplace_back(external_nn);  // for recursive
    }
    tableint next_closest_entry_point = selectedNeighbors.back();

    nns->forward_nns.emplace_back(batchNN);

    // for recursively add left points
    int recursion_counter = 0;

    vector<tableint> extend_selectedNeighbors(selectedNeighbors);

    // Recursive Add
    while (((recursion_counter++ < 100) && (!addedNeighborPositions.empty()))) {
      selectedNeighbors_set.reserve(selectedNeighbors_set.size() + M_);
      extend_selectedNeighbors.reserve(extend_selectedNeighbors.size() + M_);
      // re prune left candidates
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst>
          leftover_candidates;

      sort(addedNeighborPositions.begin(), addedNeighborPositions.end());

      switch (params->recursion_type) {
        case BaseIndex::IndexParams::Recursion_Type_t::MIN_POS:
          external_left_most = addedNeighborPositions.front();
          break;
        case BaseIndex::IndexParams::Recursion_Type_t::MID_POS:
          external_left_most =
              addedNeighborPositions[addedNeighborPositions.size() / 2];
          break;
        case BaseIndex::IndexParams::Recursion_Type_t::MAX_POS:
          external_left_most = addedNeighborPositions.back();
          break;
        case BaseIndex::IndexParams::Recursion_Type_t::SMALL_LEFT_POS:
          external_left_most = before_prune_cancidates.top().second;
      }
      // external_left_most to find the position;
      // external_left_most = addedNeighborPositions.front();
      // left most
      // left - right- most
      // middle
      // external_left_most =
      // addedNeighborPositions[addedNeighborPositions.size() / 2];

      while (!before_prune_cancidates.empty()) {
        int external_nn =
            getExternalLabel(before_prune_cancidates.top().second);
        if (external_nn > external_left_most) {
          leftover_candidates.push(before_prune_cancidates.top());
        }
        before_prune_cancidates.pop();
      }

      if (leftover_candidates.empty()) {
        // no more leftover elements, break
        // too much candidates, break
        break;
      }

      before_prune_cancidates = leftover_candidates;

      getNeighborsByHeuristic2LimitSize(leftover_candidates, M_);

      // record added neighbor to find the position
      addedNeighborPositions.clear();
      OneBatchNeighbors batchNN2(recursion_counter + 1);

      while (!leftover_candidates.empty()) {
        // successfully insert (no duplicate)?
        // do we need this for recursion methods?
        // if (selectedNeighbors_set.emplace(leftover_candidates.top().second)
        //         .second) {
        //   extend_selectedNeighbors.emplace_back(
        //       leftover_candidates.top().second);
        //   int external_nn =
        //   getExternalLabel(extend_selectedNeighbors.back());

        //   batchNN2.nns.emplace_back(
        //       make_pair(external_nn, leftover_candidates.top().first));
        //   // nns->forward_nns.emplace_back(
        //   //     make_pair(external_nn, leftover_candidates.top().first));
        //   addedNeighborPositions.emplace_back(external_nn);
        // }

        // allow duplicate

        {
          int external_nn = getExternalLabel(leftover_candidates.top().second);
          batchNN2.nns.emplace_back(
              make_pair(external_nn, leftover_candidates.top().first));
          addedNeighborPositions.emplace_back(external_nn);
        }

        leftover_candidates.pop();
      }

      batchNN2.start =
          external_left_most + 1;  // the nns id in the batch >= start position
      // record end position of last one
      nns->forward_nns.back().end = external_left_most;
      nns->forward_nns.emplace_back(batchNN2);
    }

    // // sort forward_nns, for binary search during query
    // std::sort(nns->forward_nns.begin(), nns->forward_nns.end(),
    //           [](std::pair<int, float> const &left,
    //              std::pair<int, float> const &right) {
    //             return left.first < right.first;
    //           });

    {
      linklistsizeint *ll_cur;
      ll_cur = get_linklist0(cur_c);

      if (*ll_cur && !isUpdate) {
        throw std::runtime_error(
            "The newly inserted element should have blank link list");
      }
      setListCount(ll_cur, selectedNeighbors.size());
      tableint *data = (tableint *)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        if (data[idx] && !isUpdate)
          throw std::runtime_error("Possible memory corruption");
        if (level > element_levels_[selectedNeighbors[idx]])
          throw std::runtime_error(
              "Trying to make a link on a non-existent level");

        data[idx] = selectedNeighbors[idx];
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      std::unique_lock<std::mutex> lock(
          link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint *ll_other;
      ll_other = get_linklist0(selectedNeighbors[idx]);

      size_t sz_link_list_other = getListCount(ll_other);

      if (sz_link_list_other > Mcurmax)
        throw std::runtime_error("Bad value of sz_link_list_other");
      if (selectedNeighbors[idx] == cur_c)
        throw std::runtime_error("Trying to connect an element to itself");
      if (level > element_levels_[selectedNeighbors[idx]])
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

      // move from below to here for store dist in directed_nns.
      // dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
      //                             getDataByInternalId(selectedNeighbors[idx]),
      //                             dist_func_param_);
      // add external_id to nns (backward nn)
      // int external_backward_id = getExternalLabel(selectedNeighbors[idx]);
      // directed_nns->at(external_backward_id)
      //     .reverse_nns.emplace_back(make_pair(external_id, d_max));

      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or
      // run the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);

          // // add external_id to nns (backward nn)
          // int external_backward_id =
          // getExternalLabel(selectedNeighbors[idx]);

          // directed_nns->at(external_backward_id)
          //     .reverse_nns.emplace_back(external_backward_id);

        } else {
          // finding the "weakest" element to replace it with the new one
          dist_t d_max = fstdistfunc_(
              getDataByInternalId(cur_c),
              getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
          // Heuristic:
          std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
              candidates;
          candidates.emplace(d_max, cur_c);

          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(
                fstdistfunc_(getDataByInternalId(data[j]),
                             getDataByInternalId(selectedNeighbors[idx]),
                             dist_func_param_),
                data[j]);
          }

          getNeighborsByHeuristic2(candidates, Mcurmax);

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }
          setListCount(ll_other, indx);
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

    // add reverse nns
    for (auto &batch : nns->forward_nns) {
      if (batch.nns.empty()) {
        continue;
      }

      for (auto nn : batch.nns) {
        reverse_nns_vecs.at(nn.first).emplace_back(
            make_pair(external_id, nn.second));
        batch.nns_id.emplace_back(nn.first);
      }
    }

    return next_closest_entry_point;
  }
};

class RecursionIndex : public BaseIndex {
 public:
  vector<DirectedNeighbors> directed_indexed_arr;

  RecursionIndex(delta_index_hnsw_full_reverse::SpaceInterface<float> *s,
                 const DataWrapper *data)
      : BaseIndex(data) {
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    index_info = new IndexInfo();
    index_info->index_version_type = "BaseRecursionIndex";
  }
  delta_index_hnsw_full_reverse::DISTFUNC<float> fstdistfunc_;
  void *dist_func_param_;

  VisitedListPool *visited_list_pool_;
  IndexInfo *index_info;

  // do pruning all sth else. In this base version, just no prune and collect
  // all reverse neighbor in one batch.
  void processReverseNeighbors(RangeFilteringHNSW<float> &hnsw) {
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      auto before_pruning_candidates = hnsw.reverse_nns_vecs.at(i);
      size_t batch_counter = 0;
      {
        OneBatchNeighbors batchNN(batch_counter);
        for (auto nn : before_pruning_candidates) {
          batchNN.nns_id.emplace_back(nn.first);
        }
        batchNN.start = i;
        batchNN.end = data_wrapper->data_size;
        directed_indexed_arr.at(i).reverse_nns.emplace_back(batchNN);
      }
    }
  }

  void printOnebatch() {
    for (auto nns :
         directed_indexed_arr[data_wrapper->data_size / 2].forward_nns) {
      cout << "Forward batch: " << nns.batch << "(" << nns.start << ","
           << nns.end << ")" << endl;
      print_set(nns.nns_id);
      cout << endl;
    }
    cout << endl;

    for (auto nns :
         directed_indexed_arr[data_wrapper->data_size / 2].reverse_nns) {
      cout << "Reverse batch: " << nns.batch << "(" << nns.start << ","
           << nns.end << ")" << endl;
      print_set(nns.nns_id);
      cout << endl;
    }
    cout << endl;
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
    processReverseNeighbors(hnsw);

    // count neighbors number
    countNeighbrs();

    if (index_params->print_one_batch) {
      printOnebatch();
    }
  };

  vector<OneBatchNeighbors>::const_iterator decompressForwardPath(
      const vector<OneBatchNeighbors> &forward_nns, const int lbound) {
    // forward iterator
    auto forward_batch_it = forward_nns.begin();
    while (forward_batch_it != forward_nns.end()) {
      if (lbound < forward_batch_it->end) {
        break;
      }
      forward_batch_it++;
    }
    return forward_batch_it;
  }

  vector<OneBatchNeighbors>::const_iterator decompressReversePath(
      const vector<OneBatchNeighbors> &reverse_nns, const int rbound) {
    auto reverse_batch_it = reverse_nns.begin();
    while (reverse_batch_it != reverse_nns.end()) {
      if (rbound > reverse_batch_it->start) {
        break;
      }
      reverse_batch_it++;
    }

    return reverse_batch_it;
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

        // only one batch, just insert it
        auto reverse_it =
            directed_indexed_arr[current_node_id].reverse_nns.begin();
        if (reverse_it !=
            directed_indexed_arr[current_node_id].reverse_nns.end()) {
          neighbor_iterators.emplace_back(reverse_it);
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

        auto reverse_it =
            directed_indexed_arr[current_node_id].reverse_nns.begin();
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

  ~RecursionIndex() {
    delete index_info;
    directed_indexed_arr.clear();
    delete visited_list_pool_;
  }
};
}  // namespace rangeindex