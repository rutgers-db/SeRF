
/**
 * Delta index of compact window, Using Delta Array;
 * Open Interval, One way. Directly Composite HNSW.
 * Record lifecycle of every node.
 *
 * Author: Chaoji Zuo
 * Email:  chaoji.zuo@rutgers.edu
 * Date:   Dec 4, 2022
 */
#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>

#include "../../include/lib_nndescent/NNDescent.hpp"
#include "../../include/lib_nndescent/VicDescent-Hierachically.hpp"
#include "../../include/lib_nndescent/VicDescent.hpp"
#include "../../include/lib_pq/pq.hpp"
#include "../CompositeKNNG/range_index_delta.h"
#include "../NearestSearch.h"
#include "../baselines/greedy.h"
#include "../baselines/knn_first_baseline.h"
#include "../range_index_base.h"
#include "../utils/cw.h"
#include "../utils/segmentTree.hpp"
#include "../utils/utils.hpp"
#include "../utils/utils2.h"
#include "hnswalg.h"
#include "hnswlib.h"

using std::cout;
using std::endl;
using std::vector;

namespace delta_index_hnsw_one_way {

#define SUB_NONE -1;
#define ADD_NONE -2;

struct DeltaNode {
  DeltaNode(int s, int a) : sub(s), add(a){};
  DeltaNode() {
    sub = SUB_NONE;
    add = ADD_NONE;
  };
  int sub;
  int add;
};

class DeltaIndex : public BaseIndex {
 private:
  int ltemp;
  int rtemp;

 public:
  // vector<pair<vector<DeltaNode>, vector<DeltaNode>>> indexed_arr;

  vector<vector<hnswlib_compose::NeighborLifeCycle>> indexed_arr;

  vector<vector<pair<int, vector<hnswlib_compose::NeighborLifeCycle>>>>
      indexed_arr_round;

  vector<vector<hnswlib_compose::NeighborLifeCyclePairWindows>>
      indexed_pair_windows;

 public:
  int isSparse = false;

  bool is_recursively_add = false;
  size_t range_add_type;

 public:
  DeltaIndex() {
    window_count = 0;
    nodes_amount = 0;
    sort_time = 0;
  };

  int start_;
  int end_;

  void indexNodes(const int query_pos, const vector<vector<float>> &data_nodes,
                  const vector<int> &idx_sorted,
                  const vector<vector<int>> &hnsw_idxes, const int k_smallest,
                  const int l_bound, const int r_bound,
                  vector<DeltaNode> &left_pairs, vector<DeltaNode> &right_pairs,
                  delta_index::DeltaIndex &nnIndex);

  void buildIndex(const vector<vector<float>> &nodes, const int build_knn_k,
                  const int index_K);

  void buildIndexTwoWay(const vector<vector<float>> &nodes, const int index_k,
                        const size_t build_type);

  void buildIndex(const vector<vector<float>> &nodes,
                  const vector<int> &valid_idxes, const int initial_build_k,
                  const int index_k);

  void buildIndex(const int gap, const vector<vector<float>> &nodes,
                  const int index_K);

  vector<int> indexNeighborSearch(const vector<vector<float>> &data_nodes,
                                  const vector<float> &query, const int l_bound,
                                  const int r_bound, const int K_neigbhors,
                                  const int ef);
  vector<int> indexNeighborSearch(const vector<vector<float>> &data_nodes,
                                  const vector<float> &query, const int l_bound,
                                  const int r_bound, const int K_neigbhors,
                                  const int ef,
                                  const vector<int> &enter_points);

  vector<int> indexNeighborSearch(const int gap,
                                  const vector<vector<float>> &data_nodes,
                                  const vector<float> &query, const int l_bound,
                                  const int r_bound, const int K_neigbhors,
                                  const int ef,
                                  const vector<int> &enter_points);

  vector<int> indexNeighborSearchSubGraph(
      const vector<vector<float>> &data_nodes, const vector<float> &query,
      const int l_bound, const int r_bound, const int K_neigbhors, const int ef,
      const vector<int> &enter_points);

  vector<int> indexNeighborSearchPairWindows(
      const vector<vector<float>> &data_nodes, const vector<float> &query,
      const int l_bound, const int r_bound, const int K_neigbhors, const int ef,
      const vector<int> &enter_points);

  void buildIndex(const int start, const int end,
                  const vector<vector<float>> &nodes, const int index_K);
  double calSize();
  pair<vector<int>, vector<int>> checkIndexQuality(const int idx,
                                                   const int lbound,
                                                   const int rbound);

  void clear() {
    indexed_arr.clear();
    window_count = 0;
  }
  ~DeltaIndex() { indexed_arr.clear(); }
};

// index sorted vector
// idx_sorted: index vector sorted by distance to query point
void DeltaIndex::indexNodes(const int query_pos,
                            const vector<vector<float>> &data_nodes,
                            const vector<int> &idx_sorted,
                            const vector<vector<int>> &hnsw_idxes,
                            const int k_smallest, const int l_bound,
                            const int r_bound, vector<DeltaNode> &left_pairs,
                            vector<DeltaNode> &right_pairs,
                            delta_index::DeltaIndex &nnIndex) {
  return;
}

void DeltaIndex::buildIndex(const vector<vector<float>> &nodes,
                            const int build_knn_k, const int index_K) {
  cout << "ERROR: Unimplemented Function" << endl;
}

void DeltaIndex::buildIndexTwoWay(const vector<vector<float>> &nodes,
                                  const int index_K, const size_t build_type) {
  K2_ = index_K;
  timeval tt1, tt2;
  gettimeofday(&tt1, NULL);

  // build hnsw
  hnswlib_compose::L2Space space(nodes.front().size());
  hnswlib_compose::HierarchicalNSW<float> hnsw(&space, 2 * nodes.size(),
                                               index_K, 500);
  // hnsw.maxM0_ = index_K; // no limitation
  range_add_type = build_type;
  hnsw.range_add_type = build_type;

  switch (build_type) {
    case 0:
      cout << "Wrong Type" << endl;
      break;
    case 1:
      cout << "Single Way" << endl;
      indexed_arr.resize(nodes.size());
      hnsw.range_nns = &indexed_arr;
      break;

    case 2:
      cout << "Two Way Simple" << endl;
      indexed_arr.resize(nodes.size());
      hnsw.range_nns = &indexed_arr;
      break;

    case 3:
      cout << "Two way round" << endl;
      indexed_arr_round.resize(nodes.size());
      hnsw.rounds_range_nns = &indexed_arr_round;
      hnsw.is_recursively_add = is_recursively_add;
      break;

    case 4:
      cout << "Two way pair windows" << endl;
      indexed_pair_windows.resize(nodes.size());
      hnsw.range_nns_pair = &indexed_pair_windows;
      break;
  }

  // #pragma omp parallel for
  for (size_t i = 0; i < nodes.size(); ++i) {
    hnsw.addPoint(nodes.at(i).data(), i);
  }

  int counter = 0;

  int neighbor_amount = 0;

  if (isLog) {
    logTime(tt1, tt2, "Compute HNSW graph Time");
  }

  // for (auto ele : indexed_arr[0]) {
  //   cout << "nn: " << ele.id << " (" << ele.start << "," << ele.end << ")"
  //        << endl;
  // }

  // cout << endl;
  // for (auto ele : indexed_arr[100]) {
  //   cout << "nn: " << ele.id << " (" << ele.start << "," << ele.end << ")"
  //        << endl;
  // }

  if (!indexed_arr.empty())
    for (unsigned j = 0; j < indexed_arr.size(); j++) {
      nodes_amount += indexed_arr[j].size();
    }

  if (!indexed_pair_windows.empty()) {
    for (unsigned j = 0; j < indexed_pair_windows.size(); j++) {
      nodes_amount += indexed_pair_windows[j].size();
    }
  }
  cout << "Avg. delta nn #: " << nodes_amount / (float)nodes.size() << endl;
  window_count = nodes_amount;
}

// build index on nodes[start,end].
void DeltaIndex::buildIndex(const int start, const int end,
                            const vector<vector<float>> &nodes,
                            const int index_K) {
  start_ = start;
  end_ = end;

  const int size_ = end - start + 1;
  indexed_arr.clear();
  indexed_arr.resize(size_);
  K1_ = index_K;
  K2_ = index_K;
  timeval tt1, tt2;
  gettimeofday(&tt1, NULL);
  vector<vector<int>> idxes;

  // build hnsw
  hnswlib_compose::L2Space space(nodes.front().size());
  hnswlib_compose::HierarchicalNSW<float> hnsw(&space, 2 * size_, index_K, 500);
  // hnsw.maxM0_ = index_K;
  hnsw.range_nns = &indexed_arr;
  hnsw.is_recursively_add = is_recursively_add;
  hnsw.is_two_way = false;
  // #pragma omp parallel for
  for (size_t i = start; i <= end; ++i) {
    hnsw.addPoint(nodes.at(i).data(), i - start);
  }

  int counter = 0;
  int neighbor_amount = 0;

  for (unsigned j = 0; j < indexed_arr.size(); j++) {
    nodes_amount += indexed_arr[j].size();
  }
  window_count = nodes_amount;
}

void DeltaIndex::buildIndex(const int gap, const vector<vector<float>> &nodes,
                            const int index_K) {
  indexed_arr.clear();
  indexed_arr.resize(nodes.size() - gap);

  const int size_ = nodes.size() - gap;
  K1_ = index_K;
  K2_ = index_K;
  timeval tt1, tt2;
  gettimeofday(&tt1, NULL);
  vector<vector<int>> idxes;

  // build hnsw
  hnswlib_compose::L2Space space(nodes.front().size());
  hnswlib_compose::HierarchicalNSW<float> hnsw(&space, 2 * size_, index_K, 500);
  // hnsw.maxM0_ = index_K;
  hnsw.range_nns = &indexed_arr;
  hnsw.is_recursively_add = is_recursively_add;
  // #pragma omp parallel for
  for (size_t i = gap; i < nodes.size(); ++i) {
    hnsw.addPoint(nodes.at(i).data(), i - gap);
  }

  int counter = 0;
  int neighbor_amount = 0;

  for (unsigned j = 0; j < indexed_arr.size(); j++) {
    nodes_amount += indexed_arr[j].size();
  }
  window_count = nodes_amount;

  // cout << endl << "Interval: " << gap << endl;
  // cout << "neighbors of 0: " << endl;
  // for (auto ele : indexed_arr[9000 - gap]) {
  //   cout << gap + ele.id << ":(" << gap + ele.start << "," << gap + ele.end
  //        << ")\t";
  // }

  // cout << endl;
  // cout << "neighbors of 100: " << endl;

  // for (auto ele : indexed_arr[100]) {
  //   cout << gap + ele.id << ":(" << gap + ele.start << "," << gap + ele.end
  //        << ")\t";
  // }
  // cout << endl;
}

// direction: true means from left to right, false means from right to left
vector<int> decompressDeltaPath(
    const vector<hnswlib_compose::NeighborLifeCycle> &path, const int lbound,
    const int rbound, const int pos) {
  vector<int> neighbors;
  if (lbound == 0) {
    for (auto ele : path) {
      // for lbound = 0, left interval.
      // forward neighbor, lifecycle must long enough to cover
      if (ele.id < rbound)
        if (ele.end == ele.start || ele.end >= rbound) {
          neighbors.emplace_back(ele.id);
        }
    }
  } else {
    for (auto ele : path) {
      if (ele.id > pos) {
        // backward neighbor
        if (ele.start < rbound) {
          if (ele.end == ele.start || ele.end >= rbound) {
            neighbors.emplace_back(ele.id);
          }
        }
      }

      if (ele.id < pos) {
        if (ele.start > lbound) {
          // forward neighbor, lifecycle must long enough to cover
          if (ele.end == ele.start || ele.end >= rbound) {
            neighbors.emplace_back(ele.id);
          }
        }
      }
    }
  }

  return neighbors;
}

vector<int> decompressDeltaPath(
    const vector<hnswlib_compose::NeighborLifeCyclePairWindows> &path,
    const int lbound, const int rbound, const int pos) {
  vector<int> neighbors;

  for (auto ele : path) {
    if (ele.l_start < lbound &&
        (ele.l_end > lbound || ele.l_end == ele.l_start)) {
      if (ele.r_start < rbound &&
          (ele.r_end > rbound || ele.r_start == ele.r_end)) {
        neighbors.emplace_back(ele.id);
      }
    }
  }
  return neighbors;
}

vector<int> DeltaIndex::indexNeighborSearch(
    const vector<vector<float>> &data_nodes, const vector<float> &query,
    const int l_bound, const int r_bound, const int K_neigbhors, const int ef) {
  vector<int> temp;
  return indexNeighborSearch(data_nodes, query, l_bound, r_bound, K_neigbhors,
                             ef, temp);
}

// on the fly search cw
// TODO: Verify search path result
vector<int> DeltaIndex::indexNeighborSearch(
    const vector<vector<float>> &data_nodes, const vector<float> &query,
    const int l_bound, const int r_bound, const int K_neigbhors, const int ef,
    const vector<int> &enter_points) {
  if (!indexed_pair_windows.empty()) {
    return indexNeighborSearchPairWindows(data_nodes, query, l_bound, r_bound,
                                          K_neigbhors, ef, enter_points);
  }

  unordered_map<int, bool> visited_list;
  float lower_bound = INT_MAX;
  priority_queue<pair<float, int>> top_candidates;
  priority_queue<pair<float, int>> candidate_set;
  // int query_range = r_bound - l_bound;
  // float range_scale = (float)query_range / data_nodes.size();

  // multiple enter points: 10 enter points
  vector<int> enter_list;
  if (enter_points.size() == 0) {
    int interval = (r_bound - l_bound) / 3;
    for (size_t i = 1; i < 3; i++) {
      int point = l_bound + interval * i;
      float dist = EuclideanDistance(data_nodes[point], query);
      candidate_set.push(make_pair(-dist, point));
      enter_list.emplace_back(point);
    }
    // enter_list.emplace_back(l_bound + 1);
  } else {
    for (auto point : enter_points) {
      float dist = EuclideanDistance(data_nodes[point], query);
      candidate_set.push(make_pair(-dist, point));
      enter_list.emplace_back(point);
    }
  }

  while (!candidate_set.empty()) {
    std::pair<float, int> current_node_pair = candidate_set.top();
    int current_node_id = current_node_pair.second;

    if (-current_node_pair.first > lower_bound) {
      break;
    }

    // cout << "current node: " << current_node_pair.second << "   ";
    candidate_set.pop();

    // only search when candidate point is inside the range
    if (current_node_id < l_bound || current_node_id > r_bound) {
      // cout << "no satisfied range point" << endl;
      continue;
    }

    // search cw on the fly
    vector<int> current_neighbors;

    current_neighbors = decompressDeltaPath(indexed_arr[current_node_id],
                                            l_bound, r_bound, current_node_id);
    // print_set(current_neighbors);
    for (size_t i = 0; i < current_neighbors.size(); i++) {
      int candidate_id = current_neighbors[i];
      if (!visited_list[candidate_id]) {
        visited_list[candidate_id] = true;
        float dist = EuclideanDistance(query, data_nodes[candidate_id]);
        // cout << "candidate: " << candidate_id << "  dist: " << dist <<
        // endl;
        if (top_candidates.size() < ef || lower_bound > dist) {
          candidate_set.push(make_pair(-dist, candidate_id));
          top_candidates.push(make_pair(dist, candidate_id));
          if (top_candidates.size() > ef) {
            top_candidates.pop();
          }
          if (!top_candidates.empty()) {
            lower_bound = top_candidates.top().first;
          }
        }
      }
    }
  }
  vector<int> res;
  while (top_candidates.size() > K_neigbhors) {
    top_candidates.pop();
  }
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  // cout << "tree search time: " << tree_search_time << endl;
  // cout << "neighbor search time: " << neighbor_search_time << endl;
  return res;
}

vector<int> DeltaIndex::indexNeighborSearch(
    const int gap, const vector<vector<float>> &data_nodes,
    const vector<float> &query, const int l_bound, const int r_bound,
    const int K_neigbhors, const int ef, const vector<int> &enter_points) {
  unordered_map<int, bool> visited_list;
  float lower_bound = INT_MAX;
  priority_queue<pair<float, int>> top_candidates;
  priority_queue<pair<float, int>> candidate_set;

  if (gap > indexed_arr.size()) {
    return vector<int>();
  }

  // multiple enter points: 10 enter points
  vector<int> enter_list;
  {
    int lbound = l_bound;
    if (l_bound < gap) {
      lbound = gap;
    }
    int interval = (r_bound - lbound) / 3;
    for (size_t i = 1; i < 3; i++) {
      int point = lbound + interval * i;
      float dist = EuclideanDistance(data_nodes[point], query);
      candidate_set.push(make_pair(-dist, point));
      enter_list.emplace_back(point);
    }
  }

  while (!candidate_set.empty()) {
    std::pair<float, int> current_node_pair = candidate_set.top();
    int current_node_id = current_node_pair.second;

    if (-current_node_pair.first > lower_bound) {
      break;
    }

    // cout << "current node: " << current_node_pair.second << "   ";
    candidate_set.pop();

    // only search when candidate point is inside the range
    if (current_node_id < l_bound || current_node_id > r_bound) {
      // cout << "no satisfied range point" << endl;
      continue;
    }

    // search cw on the fly
    vector<int> current_neighbors;

    current_neighbors =
        decompressDeltaPath(indexed_arr[current_node_id - gap], l_bound - gap,
                            r_bound - gap, current_node_id - gap);
    // print_set(current_neighbors);
    for (size_t i = 0; i < current_neighbors.size(); i++) {
      int candidate_id = current_neighbors[i] + gap;
      if (!visited_list[candidate_id]) {
        visited_list[candidate_id] = true;
        float dist = EuclideanDistance(query, data_nodes[candidate_id]);
        // cout << "candidate: " << candidate_id << "  dist: " << dist <<
        // endl;
        if (top_candidates.size() < ef || lower_bound > dist) {
          candidate_set.push(make_pair(-dist, candidate_id));
          top_candidates.push(make_pair(dist, candidate_id));
          if (top_candidates.size() > ef) {
            top_candidates.pop();
          }
          if (!top_candidates.empty()) {
            lower_bound = top_candidates.top().first;
          }
        }
      }
    }
  }
  vector<int> res;
  while (top_candidates.size() > K_neigbhors) {
    top_candidates.pop();
  }
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  // cout << "tree search time: " << tree_search_time << endl;
  // cout << "neighbor search time: " << neighbor_search_time << endl;
  return res;
}

vector<int> DeltaIndex::indexNeighborSearchSubGraph(
    const vector<vector<float>> &data_nodes, const vector<float> &query,
    const int l_bound, const int r_bound, const int K_neigbhors, const int ef,
    const vector<int> &enter_points) {
  unordered_map<int, bool> visited_list;
  float lower_bound = INT_MAX;
  priority_queue<pair<float, int>> top_candidates;
  priority_queue<pair<float, int>> candidate_set;

  if (l_bound > end_ || r_bound < start_) {
    return vector<int>();
  }

  // multiple enter points: 10 enter points
  vector<int> enter_list;
  {
    int lbound = l_bound;
    if (l_bound < start_) {
      lbound = start_;
    }
    int rbound = r_bound;
    if (r_bound > end_) {
      rbound = end_;
    }

    int interval = (rbound - lbound) / 3;
    for (size_t i = 1; i < 3; i++) {
      int point = lbound + interval * i;
      float dist = EuclideanDistance(data_nodes[point], query);
      candidate_set.push(make_pair(-dist, point));
      enter_list.emplace_back(point);
    }
  }

  while (!candidate_set.empty()) {
    std::pair<float, int> current_node_pair = candidate_set.top();
    int current_node_id = current_node_pair.second;

    if (-current_node_pair.first > lower_bound) {
      break;
    }

    // cout << "current node: " << current_node_pair.second << "   ";
    candidate_set.pop();

    // only search when candidate point is inside the range
    if (current_node_id < l_bound || current_node_id > r_bound) {
      // cout << "no satisfied range point" << endl;
      continue;
    }

    // search cw on the fly
    vector<int> current_neighbors;

    current_neighbors = decompressDeltaPath(
        indexed_arr[current_node_id - start_], l_bound - start_,
        r_bound - start_, current_node_id - start_);
    // print_set(current_neighbors);
    for (size_t i = 0; i < current_neighbors.size(); i++) {
      int candidate_id = current_neighbors[i] + start_;
      if (!visited_list[candidate_id]) {
        visited_list[candidate_id] = true;
        float dist = EuclideanDistance(query, data_nodes[candidate_id]);
        // cout << "candidate: " << candidate_id << "  dist: " << dist <<
        // endl;
        if (top_candidates.size() < ef || lower_bound > dist) {
          candidate_set.push(make_pair(-dist, candidate_id));
          top_candidates.push(make_pair(dist, candidate_id));
          if (top_candidates.size() > ef) {
            top_candidates.pop();
          }
          if (!top_candidates.empty()) {
            lower_bound = top_candidates.top().first;
          }
        }
      }
    }
  }
  vector<int> res;
  while (top_candidates.size() > K_neigbhors) {
    top_candidates.pop();
  }
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  // cout << "tree search time: " << tree_search_time << endl;
  // cout << "neighbor search time: " << neighbor_search_time << endl;
  return res;
}

vector<int> DeltaIndex::indexNeighborSearchPairWindows(
    const vector<vector<float>> &data_nodes, const vector<float> &query,
    const int l_bound, const int r_bound, const int K_neigbhors, const int ef,
    const vector<int> &enter_points) {
  unordered_map<int, bool> visited_list;
  float lower_bound = INT_MAX;
  priority_queue<pair<float, int>> top_candidates;
  priority_queue<pair<float, int>> candidate_set;

  // multiple enter points: 10 enter points
  vector<int> enter_list;
  if (enter_points.size() == 0) {
    int interval = (r_bound - l_bound) / 3;
    for (size_t i = 1; i < 3; i++) {
      int point = l_bound + interval * i;
      float dist = EuclideanDistance(data_nodes[point], query);
      candidate_set.push(make_pair(-dist, point));
      enter_list.emplace_back(point);
    }
    // enter_list.emplace_back(l_bound + 1);
  } else {
    for (auto point : enter_points) {
      float dist = EuclideanDistance(data_nodes[point], query);
      candidate_set.push(make_pair(-dist, point));
      enter_list.emplace_back(point);
    }
  }

  while (!candidate_set.empty()) {
    std::pair<float, int> current_node_pair = candidate_set.top();
    int current_node_id = current_node_pair.second;

    if (-current_node_pair.first > lower_bound) {
      break;
    }

    // cout << "current node: " << current_node_pair.second << "   ";
    candidate_set.pop();

    // only search when candidate point is inside the range
    if (current_node_id < l_bound || current_node_id > r_bound) {
      // cout << "no satisfied range point" << endl;
      continue;
    }

    // search cw on the fly
    vector<int> current_neighbors;

    current_neighbors =
        decompressDeltaPath(indexed_pair_windows[current_node_id], l_bound,
                            r_bound, current_node_id);
    // print_set(current_neighbors);
    for (size_t i = 0; i < current_neighbors.size(); i++) {
      int candidate_id = current_neighbors[i];
      if (!visited_list[candidate_id]) {
        visited_list[candidate_id] = true;
        float dist = EuclideanDistance(query, data_nodes[candidate_id]);
        // cout << "candidate: " << candidate_id << "  dist: " << dist <<
        // endl;
        if (top_candidates.size() < ef || lower_bound > dist) {
          candidate_set.push(make_pair(-dist, candidate_id));
          top_candidates.push(make_pair(dist, candidate_id));
          if (top_candidates.size() > ef) {
            top_candidates.pop();
          }
          if (!top_candidates.empty()) {
            lower_bound = top_candidates.top().first;
          }
        }
      }
    }
  }
  vector<int> res;
  while (top_candidates.size() > K_neigbhors) {
    top_candidates.pop();
  }
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  // cout << "tree search time: " << tree_search_time << endl;
  // cout << "neighbor search time: " << neighbor_search_time << endl;
  return res;
}

double DeltaIndex::calSize() {
  // double unit_size = sizeof(this->indexed_arr.front().first.front());
  // cout << "unit size: " << unit_size << endl;
  return (double)window_count;
}

pair<vector<int>, vector<int>> DeltaIndex::checkIndexQuality(const int idx,
                                                             const int lbound,
                                                             const int rbound) {
  vector<int> lres, rres;
  // lres = decompressDeltaPath(indexed_arr[idx].first, lbound, K2_, true);
  // rres = decompressDeltaPath(indexed_arr[idx].second, rbound, K2_, false);
  // // for (auto ele : indexed_arr[idx].second) {
  // //   cout << ele.add << "," << ele.sub << endl;
  // // }
  // sort(lres.begin(), lres.end());
  // sort(rres.begin(), rres.end());
  // print_set(lres);
  // print_set(rres);

  return make_pair(lres, rres);
}

}  // namespace delta_index_hnsw_one_way