/**
 * @file index_recursion_batch.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Index for arbitrary range filtering search
 * Compress N SegmentGraph
 * @date 2023-06-19; Revised 2024-01-10
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <boost/functional/hash.hpp>
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

namespace SeRF {

struct OneSegmentNeighbors {
  OneSegmentNeighbors() { batch = 0; }
  OneSegmentNeighbors(unsigned num) : batch(num) {}
  OneSegmentNeighbors(unsigned num, int start, int end)
      : batch(num), start(start), end(end) {}

  // vector<pair<int, float>> nns;
  vector<int> nns_id;
  unsigned batch;  // batch id
  int start = -1;  // left position
  int end = -2;    // right position
  const unsigned size() const { return nns_id.size(); }
};

// struct OneTupleNeighbor {
//   int start = -1;
//   int end = -2;
//   int nn_id;
// }

struct DirectedSegNeighbors {
  vector<OneSegmentNeighbors> forward_nns;
  vector<int> reverse_nns;
};

template <typename dist_t>
class SegmentGraph2DHNSW : public HierarchicalNSW<float> {
 public:
  SegmentGraph2DHNSW(const BaseIndex::IndexParams &index_params,
                     SpaceInterface<float> *s, size_t max_elements,
                     size_t M = 16, size_t ef_construction = 200,
                     size_t random_seed = 100)
      : HierarchicalNSW(s, max_elements, M, index_params.ef_construction,
                        random_seed) {
    params = &index_params;
    ef_max_ = index_params.ef_max;
  }

  const BaseIndex::IndexParams *params;
  vector<DirectedSegNeighbors> *segment_graph;

  // Rewrite the searching while construting HNSW, keep more neighbors.
  // TODO: maybe can be updated to a position sensing functions.
  // or combine with rnn-descent
  virtual std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
  searchBaseLayerLevel0(tableint ep_id, const void *data_point, int layer) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidateSet;

    std::vector<pair<dist_t, tableint>> deleted_list;

    size_t ef_construction = ef_max_;

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
      dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                 dist_func_param_);
      top_candidates.emplace(dist, ep_id);
      lowerBound = dist;
      candidateSet.emplace(-dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
      candidateSet.emplace(-lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
      if ((-curr_el_pair.first) > lowerBound) {
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second;

      std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

      int *data;  // = (int *)(linkList0_ + curNodeNum *
                  // size_links_per_element0_);
      if (layer == 0) {
        data = (int *)get_linklist0(curNodeNum);
      } else {
        data = (int *)get_linklist(curNodeNum, layer);
        //                    data = (int *) (linkLists_[curNodeNum] + (layer
        //                    - 1) * size_links_per_element_);
      }
      size_t size = getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag) continue;
        visited_array[candidate_id] = visited_array_tag;
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates.size() < ef_construction || lowerBound > dist1) {
          candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                       _MM_HINT_T0);
#endif

          if (!isMarkedDeleted(candidate_id))
            top_candidates.emplace(dist1, candidate_id);

          // record deleted neighbors
          if (top_candidates.size() > ef_construction) {
            deleted_list.emplace_back(top_candidates.top());
            top_candidates.pop();
          }

          if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    // add back deleted neighbors for recursively pruning
    for (auto deleted_candidate : deleted_list) {
      top_candidates.emplace(deleted_candidate);
    }

    return top_candidates;
  }

  // rewrite connect new element, integrate recursively heuristic pruning
  virtual tableint mutuallyConnectNewElementLevel0(
      const void *data_point, tableint cur_c,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      int level, bool isUpdate) {
    size_t Mcurmax = maxM0_;

    // forward neighbors in top candidates
    int external_id = getExternalLabel(cur_c);
    tableint next_closest_entry_point = 0;

    // original structure for fetching nodes
    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    {
      // MAX_POS recursive pruning, combining into connect function.

      // reverse top_candidates, from cloest to farthest
      std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
      while (top_candidates.size() > 0) {
        queue_closest.emplace(-top_candidates.top().first,
                              top_candidates.top().second);
        top_candidates.pop();
      }
      // Now top_candidates is empty

      int external_left_most = -1;
      int last_batch_left_most = -1;

      unsigned iter_counter = 0;
      unsigned batch_counter = 0;
      std::vector<std::pair<dist_t, tableint>> return_list;
      std::vector<int> return_external_list;

      // Need a buffer candidates to store neighbors (external_id, dist_t,
      // internal_id)
      vector<pair<int, pair<dist_t, tableint>>> buffer_candidates;

      using tableint_pair = std::pair<tableint, tableint>;
      std::unordered_map<tableint_pair, dist_t, boost::hash<tableint_pair>>
          visited_nn_dists;

      while (queue_closest.size()) {
        // If return list size meet M or current window exceed ef_construction,
        // end this batch, enter the new batch
        if (return_list.size() >= Mcurmax ||
            iter_counter >= ef_basic_construction_) {
          // reset batch, add current batch;
          // no breaking because recursivly visiting the candidates.

          if (batch_counter == 0) {
            // The first batch, also use for original HNSW constructing
            next_closest_entry_point =
                return_list.front()
                    .second;  // TODO: check whether the nearest neighbor
            for (std::pair<dist_t, tableint> curent_pair : return_list) {
              selectedNeighbors.push_back(curent_pair.second);
            }
          }

          // TODO: For MID_POS and MIN_POS, find the external_left_most by
          // sorting.
          // const auto [min_pos, max_pos] = std::minmax_element(
          //     std::begin(return_external_list),
          //     std::end(return_external_list));
          // assert(*max_pos == external_left_most);
          // OneSegmentNeighbors one_segment(batch_counter, *min_pos, *max_pos);

          for (pair<int, pair<dist_t, tableint>> curent_buffer :
               buffer_candidates) {
            if (curent_buffer.first > external_left_most) {
              // available in next batch, add back to the queue.
              queue_closest.emplace(curent_buffer.second);
            }
          }

          // current segment start position: [last left + 1, current most]
          OneSegmentNeighbors one_segment(
              batch_counter, last_batch_left_most + 1, external_left_most);

          // for (size_t p_idx = 0; p_idx < return_list.size(); p_idx++) {
          //   // keep dists
          //   // one_segment.nns.emplace_back(return_external_list.at(p_idx),
          //   //                              -return_list.at(p_idx).first);
          // }
          // only keep id, drop dists
          one_segment.nns_id.swap(return_external_list);
          segment_graph->at(external_id).forward_nns.emplace_back(one_segment);

          return_list.clear();
          return_external_list.clear();
          iter_counter = 0;
          batch_counter++;
          last_batch_left_most = external_left_most;
        }

        // experimets show that this doesn't matter a lot
        // if queue smaller than Mcurmax, add all
        // if ((return_list.size() + queue_closest.size()) < Mcurmax) {
        //   while (queue_closest.size()) {
        //     std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        //     int curent_external_id = getExternalLabel(curent_pair.second);
        //     queue_closest.pop();
        //     if (curent_external_id > external_left_most) {
        //       return_list.push_back(curent_pair);
        //       return_external_list.push_back(curent_external_id);
        //     }
        //   }
        //   break;
        // }

        std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        dist_t dist_to_query = -curent_pair.first;
        queue_closest.pop();

        int curent_external_id = getExternalLabel(curent_pair.second);
        if (curent_external_id < last_batch_left_most) {
          // position left than last batch, skip
          continue;
        }
        iter_counter++;
        bool good = true;

        for (std::pair<dist_t, tableint> second_pair : return_list) {
          dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);

          // use a map structure to record the distances, maybe can boost
          // MID_POS and MIN_POS
          // dist_t curdist; tableint_pair nn_pair =
          //     std::make_pair(std::min(second_pair.second,
          //     curent_pair.second),
          //                    std::max(second_pair.second,
          //                    curent_pair.second));
          // if (visited_nn_dists.count(nn_pair)) {
          //   curdist = visited_nn_dists.at(nn_pair);
          // } else {
          //   curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
          //                          getDataByInternalId(curent_pair.second),
          //                          dist_func_param_);
          //   visited_nn_dists.insert({nn_pair, curdist});
          // }

          if (curdist < dist_to_query) {
            good = false;
            break;
          }
        }
        if (good) {
          return_list.push_back(curent_pair);
          return_external_list.push_back(curent_external_id);
          if (curent_external_id > external_left_most) {
            external_left_most = curent_external_id;
          }
        } else {
          // decide whether add to buffer list, TODO: consider MIN_POS, MID_POS
          if (curent_external_id > external_left_most) {
            buffer_candidates.emplace_back(
                make_pair(curent_external_id, curent_pair));
          }
        }
      }

      if (batch_counter == 0) {
        // The first batch, also use for original HNSW constructing
        next_closest_entry_point = return_list.front().second;
        for (std::pair<dist_t, tableint> curent_pair : return_list) {
          selectedNeighbors.push_back(curent_pair.second);
        }
      }
      if (!return_list.empty()) {
        OneSegmentNeighbors one_segment(batch_counter, last_batch_left_most + 1,
                                        external_id);
        // for (size_t p_idx = 0; p_idx < return_list.size(); p_idx++) {
        //   // keep dists
        //   // one_segment.nns.emplace_back(return_external_list.at(p_idx),
        //   //                              -return_list.at(p_idx).first);
        // }
        // only keep id, drop dists
        one_segment.nns_id.swap(return_external_list);
        segment_graph->at(external_id).forward_nns.emplace_back(one_segment);
      }
    }

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

      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or
      // run the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);

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
          // int indx = -1;
          // for (int j = 0; j < sz_link_list_other; j++) {
          //     dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
          // getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
          //         indx = j;
          //         d_max = d;
          //     }
          // }
          // if (indx >= 0) {
          //     data[indx] = cur_c;
          // }
        }
      }
    }

    // Update: integrate this step in processing reverse edges.
    // just add reverse nns, no pruning, no processing
    // for (auto &batch : segment_graph->at(external_id).forward_nns) {
    //   if (batch.nns.empty()) {
    //     continue;
    //   }

    //   for (auto nn : batch.nns) {
    //     // reverse_nns_vecs.at(nn.first).emplace_back(
    //     //     make_pair(external_id, nn.second));
    //     batch.nns_id.emplace_back(nn.first);  // TODO: put this during
    //     pruning
    //   }
    // }

    return next_closest_entry_point;
  }
};

class IndexSegmentGraph2D : public BaseIndex {
 public:
  vector<DirectedSegNeighbors> directed_indexed_arr;

  IndexSegmentGraph2D(base_hnsw::SpaceInterface<float> *s,
                      const DataWrapper *data)
      : BaseIndex(data) {
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    index_info = new IndexInfo();
    index_info->index_version_type = "IndexSegmentGraph2D";
  }
  base_hnsw::DISTFUNC<float> fstdistfunc_;
  void *dist_func_param_;

  VisitedListPool *visited_list_pool_;
  IndexInfo *index_info;
  const BaseIndex::IndexParams *index_params_;

  // connect reverse neighbors, do pruning all sth else. In this base version,
  // just no prune and collect all reverse neighbor in one batch.
  void processReverseNeighbors(SegmentGraph2DHNSW<float> &hnsw) {
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      for (auto batch : this->directed_indexed_arr.at(i).forward_nns) {
        for (auto nn : batch.nns_id) {
          // add reverse edge
          // this->directed_indexed_arr.at(nn).reverse_nns.emplace_back(i);
          this->directed_indexed_arr.at(nn).reverse_nns.insert(
              this->directed_indexed_arr.at(nn).reverse_nns.begin(), i);
        }
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

    // for (auto nns :
    //      directed_indexed_arr[data_wrapper->data_size / 2].reverse_nns) {
    //   cout << "Reverse batch: " << nns.batch << "(" << nns.start << ","
    //        << nns.end << ")" << endl;
    //   print_set(nns.nns_id);
    //   cout << endl;
    // }

    cout << "Reverse batch: " << endl;
    print_set(directed_indexed_arr[data_wrapper->data_size / 2].reverse_nns);
    cout << endl;

    cout << endl;
  }

  void countNeighbrs() {
    double batch_counter = 0;
    double max_batch_counter = 0;
    size_t max_reverse_nn = 0;
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
        reverse_node_amount += directed_indexed_arr[j].reverse_nns.size();
        batch_counter += 1;
        max_reverse_nn = std::max(max_reverse_nn,
                                  directed_indexed_arr[j].reverse_nns.size());
      }
    }

    index_info->nodes_amount += reverse_node_amount;
    index_info->avg_reverse_nns =
        reverse_node_amount / (float)data_wrapper->data_size;
    if (isLog) {
      cout << "Max. reverse nn #: " << max_reverse_nn << endl;
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
    visited_list_pool_ =
        new base_hnsw::VisitedListPool(1, data_wrapper->data_size);

    index_params_ = index_params;
    // build HNSW
    L2Space space(data_wrapper->data_dim);
    SegmentGraph2DHNSW<float> hnsw(
        *index_params, &space, 2 * data_wrapper->data_size, index_params->K,
        index_params->ef_construction, index_params->random_seed);

    directed_indexed_arr.clear();
    directed_indexed_arr.resize(data_wrapper->data_size);
    hnsw.segment_graph = &directed_indexed_arr;
    gettimeofday(&tt1, NULL);

    // multi-thread also work, but not guaranteed as the paper
    // may has minor recall decrement
    // #pragma omp parallel for schedule(monotonic : dynamic)
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

  vector<OneSegmentNeighbors>::const_iterator decompressForwardPath(
      const vector<OneSegmentNeighbors> &forward_nns, const int lbound) {
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

  vector<OneSegmentNeighbors>::const_iterator decompressReversePath(
      const vector<OneSegmentNeighbors> &reverse_nns, const int rbound) {
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
    float lower_bound = std::numeric_limits<float>::max();
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

    // only one center
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
      vector<const vector<int> *> neighbor_iterators;

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
          neighbor_iterators.emplace_back(&forward_it->nns_id);
          if (current_node_id - query_bound.first < two_batch_threshold) {
            forward_it++;
            if (forward_it !=
                directed_indexed_arr[current_node_id].forward_nns.end()) {
              neighbor_iterators.emplace_back(&forward_it->nns_id);
            }
          }
        }

        // only one batch, just insert it
        // auto reverse_it =
        //     directed_indexed_arr[current_node_id].reverse_nns.begin();
        // if (reverse_it !=
        //     directed_indexed_arr[current_node_id].reverse_nns.end()) {
        //   neighbor_iterators.emplace_back(reverse_it);
        // }

        // Update: update the reverse structure
        neighbor_iterators.emplace_back(
            &directed_indexed_arr.at(current_node_id).reverse_nns);
      }

      gettimeofday(&tt2, NULL);
      AccumulateTime(tt1, tt2, search_info->fetch_nns_time);
      // print_set(current_neighbors);
      // assert(false);
      gettimeofday(&tt1, NULL);

      for (auto batch_it : neighbor_iterators) {
        for (auto candidate_id : *batch_it) {
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
      // vector<vector<OneSegmentNeighbors>::const_iterator> neighbor_iterators;
      vector<const vector<int> *> neighbor_iterators;

      gettimeofday(&tt1, NULL);
      {
        auto forward_it = decompressForwardPath(
            directed_indexed_arr[current_node_id].forward_nns,
            query_bound.first);
        if (forward_it !=
            directed_indexed_arr[current_node_id].forward_nns.end()) {
          neighbor_iterators.emplace_back(&forward_it->nns_id);
        }
        // Update: update the reverse structure
        neighbor_iterators.emplace_back(
            &directed_indexed_arr.at(current_node_id).reverse_nns);
      }

      gettimeofday(&tt2, NULL);
      AccumulateTime(tt1, tt2, search_info->fetch_nns_time);
      gettimeofday(&tt1, NULL);

      for (auto batch_it : neighbor_iterators) {
        unsigned visited_nn_num = 0;
        for (auto candidate_id : *batch_it) {
          if (candidate_id < query_bound.first ||
              candidate_id > query_bound.second)
            continue;
          visited_nn_num++;

          // visiting more than K neighbors, break. for reverse nn
          if (visited_nn_num > 2 * index_params_->K) break;
          if (!(visited_array[candidate_id] == visited_array_tag)) {
            visited_array[candidate_id] = visited_array_tag;

            float dist = fstdistfunc_(query.data(),
                                      data_wrapper->nodes[candidate_id].data(),
                                      dist_func_param_);

            num_search_comparison++;
            if (top_candidates.size() < search_params->search_ef ||
                lower_bound > dist) {
              candidate_set.emplace(-dist, candidate_id);
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

  ~IndexSegmentGraph2D() {
    delete index_info;
    directed_indexed_arr.clear();
    delete visited_list_pool_;
  }
};
}  // namespace SeRF