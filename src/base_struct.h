#include <vector>

using std::pair;
using std::vector;
#pragma once
// // id and dist
// // store all the information in the vector
// struct DirectedNeighbors {
//   DirectedNeighbors() {}
//   vector<pair<int, float>> forward_nns;
//   vector<pair<int, float>> reverse_nns;
// };

// a batch of neighbors having the same life cycle
struct OneBatchNeighbors {
  OneBatchNeighbors(unsigned num) : batch(num) {}
  vector<pair<int, float>> nns;
  vector<int> nns_id;
  bool is_faward;
  int start = -1;  // left position
  int end = -2;    // right position
  unsigned batch;

  void reverse_add(const pair<int, float> &node) {
    if (start == -1 || node.first < start) {
      start = node.first;
    }
    if (end == -2 || node.first > end) {
      end = node.first;
    }
    nns.emplace_back(node);
  }
  const unsigned size() const { return nns.size(); }
};

// use lifecycle to record
struct DirectedNeighbors {
  vector<OneBatchNeighbors> forward_nns;
  vector<OneBatchNeighbors> reverse_nns;
};

template <typename dist_t>
struct NeighborLifeCycle {
  NeighborLifeCycle(int a) : id(a){};
  NeighborLifeCycle(int a, dist_t b, int c, int d)
      : id(a), dist(b), start(c), end(d){};
  int id;
  dist_t dist;
  int start;
  int end;
};

struct NeighborLifeCyclePairWindows {
  NeighborLifeCyclePairWindows(int a, float dist_) : id(a), dist(dist_){};
  NeighborLifeCyclePairWindows(int a, int b, int c, int d, int e)
      : id(a), l_begin(b), l_end(c), r_begin(d), r_end(e){};
  NeighborLifeCyclePairWindows(int a, int b, int c, int d, int e, float dist_)
      : id(a), l_begin(b), l_end(c), r_begin(d), r_end(e), dist(dist_){};
  NeighborLifeCyclePairWindows(int a, int b, int c, float dist_)
      : id(a), l_begin(b), l_end(c), dist(dist_){};
  int id;
  float dist;
  int l_begin;
  int l_end;
  int r_begin;
  int r_end;
  bool is_close = false;
};

struct DirectedNeighborsPairWindows {
  vector<NeighborLifeCyclePairWindows> forward_nns;
  vector<NeighborLifeCyclePairWindows> reverse_nns;
};