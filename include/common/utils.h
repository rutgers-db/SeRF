/**
 * @file utils.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Utils Functions
 * @date 2023-04-21
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include <assert.h>
#include <sys/time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#elif __APPLE__
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#endif

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;

float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs,
                        const int &startDim, int lensDim);

float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs);

float EuclideanDistanceSquare(const vector<float> &lhs,
                              const vector<float> &rhs);

void AccumulateTime(timeval &t2, timeval &t1, double &val_time);
void CountTime(timeval &t1, timeval &t2, double &val_time);
double CountTime(timeval &t1, timeval &t2);

// the same to sort_indexes
template <typename T>
std::vector<std::size_t> sort_permutation(const std::vector<T> &vec) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
            [&](std::size_t i, std::size_t j) { return vec[i] < vec[j]; });
  return p;
}

// apply permutation
template <typename T>
void apply_permutation_in_place(std::vector<T> &vec,
                                const std::vector<std::size_t> &p) {
  std::vector<bool> done(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    if (done[i]) {
      continue;
    }
    done[i] = true;
    std::size_t prev_j = i;
    std::size_t j = p[i];
    while (i != j) {
      std::swap(vec[prev_j], vec[j]);
      done[j] = true;
      prev_j = j;
      j = p[j];
    }
  }
}

template <typename T>
vector<int> sort_indexes(const vector<T> &v) {
  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

template <typename T>
vector<int> sort_indexes(const vector<T> &v, const int begin_bias,
                         const int end_bias) {
  // initialize original index locations
  vector<int> idx(end_bias - begin_bias);
  iota(idx.begin() + begin_bias, idx.begin() + end_bias, 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin() + begin_bias, idx.begin() + end_bias,
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

template <typename T>
void print_set(const vector<T> &v) {
  if (v.size() == 0) {
    cout << "ERROR: EMPTY VECTOR!" << endl;
    return;
  }
  cout << "vertex in set: {";
  for (size_t i = 0; i < v.size() - 1; i++) {
    cout << v[i] << ", ";
  }
  cout << v.back() << "}" << endl;
}

void logTime(timeval &begin, timeval &end, const string &log);

double countPrecision(const vector<int> &truth, const vector<int> &pred);
double countApproximationRatio(const vector<vector<float>> &raw_data,
                               const vector<int> &truth,
                               const vector<int> &pred,
                               const vector<float> &query);

void print_memory();
void record_memory(long long &);
#define _INT_MAX 2147483640

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int k_smallest);

// void evaluateKNNG(const vector<vector<int>> &gt,
//                   const vector<vector<int>> &knng, const int K, double
//                   &recall, double &precision);

void rangeGreedy(const vector<vector<float>> &nodes, const int k_smallest,
                 const int l_bound, const int r_bound);

void greedyNearest(const int query_pos, const vector<vector<float>> &dpts,
                   const int k_smallest, const int l_bound, const int r_bound);

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int l_bound,
                          const int r_bound, const int k_smallest);

void heuristicPrune(const vector<vector<float>> &nodes,
                    vector<pair<int, float>> &top_candidates, const size_t M);

vector<int> str2vec(const string str);