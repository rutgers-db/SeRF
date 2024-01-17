/**
 * @file segment_graph_1d.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Index for half-bounded range filtering search.
 * Lossless compression on N hnsw on search space
 * @date 2023-06-29; Revised 2023-12-29
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "utils.h"

using namespace base_hnsw;
// #define INT_MAX __INT_MAX__

namespace SeRF {

/**
 * @brief segment neighbor structure to store segment graph edge information
 * if id==end_id, means haven't got pruned
 * id: neighbor id
 * dist: neighbor dist
 * end_id: when got pruned
 */
template <typename dist_t>
struct SegmentNeighbor1D {
  SegmentNeighbor1D(int id) : id(id){};
  SegmentNeighbor1D(int id, dist_t dist, int end_id)
      : id(id), dist(dist), end_id(end_id){};
  int id;
  dist_t dist;
  int end_id;
};

// Inherit from basic HNSW, modify the 'heuristic pruning' procedure to record
// the lifecycle for SegmentGraph
template <typename dist_t>
class SegmentGraph1DHNSW : public HierarchicalNSW<float> {
 public:
  SegmentGraph1DHNSW(const BaseIndex::IndexParams &index_params,
                     SpaceInterface<dist_t> *s, size_t max_elements,
                     size_t M = 16, size_t ef_construction = 200,
                     size_t random_seed = 100)
      : HierarchicalNSW(s, max_elements, M, index_params.ef_construction,
                        random_seed) {
    params = &index_params;
    // in ons-side segment graph, ef_max_ equal to ef_construction
    ef_max_ = index_params.ef_construction;
    ef_basic_construction_ = index_params.ef_construction;
    ef_construction = index_params.ef_construction;
  }

  const BaseIndex::IndexParams *params;

  // index storing structure
  vector<vector<SegmentNeighbor1D<dist_t>>> *range_nns;

  void getNeighborsByHeuristic2RecordPruned(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      const size_t M, vector<SegmentNeighbor1D<dist_t>> *back_nns,
      const int end_pos_id) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first,
                            top_candidates.top().second);
      top_candidates.pop();
    }

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
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      } else {
        // record pruned nns, store in range_nns
        int external_nn = this->getExternalLabel(curent_pair.second);
        if (external_nn != end_pos_id) {
          SegmentNeighbor1D<dist_t> pruned_nn(external_nn, dist_to_query,
                                              end_pos_id);
          back_nns->emplace_back(pruned_nn);
        }
      }
    }

    // add unvisited nns
    while (queue_closest.size()) {
      std::pair<dist_t, tableint> curent_pair = queue_closest.top();
      int external_nn = this->getExternalLabel(curent_pair.second);
      queue_closest.pop();

      if (external_nn != end_pos_id) {
        SegmentNeighbor1D<dist_t> pruned_nn(external_nn, -curent_pair.first,
                                            end_pos_id);
        back_nns->emplace_back(pruned_nn);
      }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
  }

  // since the order is important, SeRF use the external_id rather than the
  // inernal_id, but right now SeRF only supports building in one thread, so
  // acutally current external_id is equal to internal_id(cur_c).
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

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(this->M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);

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

          // TODO: add mutex to support parallel
          // auto back_nns = &range_nns->at(selectedNeighbors[idx]);
          auto back_nns =
              &range_nns->at(this->getExternalLabel(selectedNeighbors[idx]));
          getNeighborsByHeuristic2RecordPruned(candidates, Mcurmax, back_nns,
                                               external_id);
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
class IndexSegmentGraph1D : public BaseIndex {
 public:
  vector<vector<SegmentNeighbor1D<dist_t>>> indexed_arr;

  IndexSegmentGraph1D(base_hnsw::SpaceInterface<dist_t> *s,
                      const DataWrapper *data)
      : BaseIndex(data) {
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    index_info = new IndexInfo();
    index_info->index_version_type = "IndexSegmentGraph1D";
  }

  base_hnsw::DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_;

  VisitedListPool *visited_list_pool_;
  IndexInfo *index_info;

  void printOnebatch(int pos = -1) {
    if (pos == -1) {
      pos = data_wrapper->data_size / 2;
    }

    cout << "nns at position: " << pos << endl;
    for (auto nns : indexed_arr[pos]) {
      cout << nns.id << "->" << nns.end_id << ")\n" << endl;
    }
    cout << endl;
  }

  void countNeighbors() {
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
    }

    index_info->avg_reverse_nns = 0;
  }

  void buildIndex(const IndexParams *index_params) override {
    cout << "Building Index using " << index_info->index_version_type << endl;
    timeval tt1, tt2;
    visited_list_pool_ =
        new base_hnsw::VisitedListPool(1, data_wrapper->data_size);

    // build HNSW
    L2Space space(data_wrapper->data_dim);
    SegmentGraph1DHNSW<float> hnsw(
        *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
        index_params->ef_construction, index_params->random_seed);

    indexed_arr.clear();
    indexed_arr.resize(data_wrapper->data_size);

    hnsw.range_nns = &indexed_arr;

    gettimeofday(&tt1, NULL);

    // multi-thread also work, but not guaranteed as the paper
    // may has minor recall decrement
    // #pragma omp parallel for schedule(monotonic : dynamic)
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      // cout << i << endl;
      hnsw.addPoint(data_wrapper->nodes.at(i).data(), i);
    }

    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      // insert not pruned hnsw graph back
      hnsw.get_linklist0(i);
      linklistsizeint *ll_cur;
      ll_cur = hnsw.get_linklist0(i);
      size_t link_list_count = hnsw.getListCount(ll_cur);
      tableint *data = (tableint *)(ll_cur + 1);

      for (size_t j = 0; j < link_list_count; j++) {
        int node_id = hnsw.getExternalLabel(data[j]);
        SegmentNeighbor1D<dist_t> nn(node_id, 0, node_id);
        indexed_arr.at(i).emplace_back(nn);
      }
    }
    logTime(tt1, tt2, "Construct Time");
    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);
    // count neighbors number
    countNeighbors();

    if (index_params->print_one_batch) {
      printOnebatch();
    }
  }

  // range filtering search, only calculate distance on on-range nodes.
  vector<int> rangeFilteringSearchInRange(
      const SearchParams *search_params, SearchInfo *search_info,
      const vector<float> &query,
      const std::pair<int, int> query_bound) override {
    // timeval tt1, tt2;
    timeval tt3, tt4;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    float lower_bound = std::numeric_limits<float>::max();
    std::priority_queue<pair<float, int>> top_candidates;
    std::priority_queue<pair<float, int>> candidate_set;

    search_info->total_comparison = 0;
    search_info->internal_search_time = 0;
    search_info->cal_dist_time = 0;
    search_info->fetch_nns_time = 0;
    num_search_comparison = 0;
    gettimeofday(&tt3, NULL);

    // three enters, SeRF disgard the hierarchical structure of HNSW
    vector<int> enter_list;
    {
      int lbound = query_bound.first;
      int interval = (query_bound.second - lbound) / 3;
      for (size_t i = 0; i < 3; i++) {
        int point = lbound + interval * i;
        float dist = fstdistfunc_(
            query.data(), data_wrapper->nodes[point].data(), dist_func_param_);
        candidate_set.push(make_pair(-dist, point));
        enter_list.emplace_back(point);
        visited_array[point] = visited_array_tag;
      }
    }

    // size_t hop_counter = 0;

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

      // hop_counter++;
      candidate_set.pop();
      auto neighbor_it = indexed_arr.at(current_node_id).begin();

      // gettimeofday(&tt1, NULL);

      while (neighbor_it != indexed_arr[current_node_id].end()) {
        if ((neighbor_it->id < query_bound.second) &&
            (neighbor_it->end_id == neighbor_it->id ||
             neighbor_it->end_id >= query_bound.second)) {
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
      // gettimeofday(&tt2, NULL);
      // AccumulateTime(tt1, tt2, search_info->cal_dist_time);
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
    // timeval tt1, tt2;
    timeval tt3, tt4;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    float lower_bound = std::numeric_limits<float>::max();
    std::priority_queue<pair<float, int>> top_candidates;
    std::priority_queue<pair<float, int>> candidate_set;

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
        float dist = fstdistfunc_(
            query.data(), data_wrapper->nodes[point].data(), dist_func_param_);
        candidate_set.push(make_pair(-dist, point));
        enter_list.emplace_back(point);
        visited_array[point] = visited_array_tag;
      }
    }
    gettimeofday(&tt3, NULL);

    // size_t hop_counter = 0;

    while (!candidate_set.empty()) {
      std::pair<float, int> current_node_pair = candidate_set.top();
      int current_node_id = current_node_pair.second;
      if (-current_node_pair.first > lower_bound) {
        break;
      }

      // hop_counter++;

      candidate_set.pop();

      auto neighbor_it = indexed_arr.at(current_node_id).begin();
      // gettimeofday(&tt1, NULL);

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

      // gettimeofday(&tt2, NULL);
      // AccumulateTime(tt1, tt2, search_info->cal_dist_time);
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

  // TODO: save and load segment graph 1d
  void save(const string &save_path) {}

  void load(const string &load_path) {}

  ~IndexSegmentGraph1D() {
    delete index_info;
    indexed_arr.clear();
    delete visited_list_pool_;
  }
};

}  // namespace SeRF