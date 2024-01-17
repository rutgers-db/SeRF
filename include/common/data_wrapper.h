/**
 * @file data_vecs.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Control the raw vector and querys
 * @date 2023-06-19
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include <string>
#include <vector>

using std::pair;
using std::string;
using std::vector;

class DataWrapper {
 public:
  DataWrapper(int num, int k_, string dataset_name, int data_size_)
      : dataset(dataset_name),
        data_size(data_size_),
        query_num(num),
        query_k(k_){};
  const string dataset;
  string version;
  const int data_size;
  const int query_num;

  const int query_k;
  size_t data_dim;

  bool is_even_weight;
  bool real_keys;

  // TODO: change vector storage to array
  vector<vector<float>> nodes;
  vector<int> nodes_keys;  // search_keys
  vector<vector<float>>
      querys;  // raw querys; less than query_ids and query_ranges;
  vector<int> querys_keys;
  vector<pair<int, int>> query_ranges;
  vector<vector<int>> groundtruth;
  vector<int> query_ids;
  void readData(string &dataset_path, string &query_path);
  void generateRangeFilteringQueriesAndGroundtruth(bool is_save = false,
                                                   const string path = "");
  void generateHalfBoundedQueriesAndGroundtruth(bool is_save = false,
                                                const string path = "");
  void LoadGroundtruth(const string &gt_path);

  void generateRangeFilteringQueriesAndGroundtruthScalability(
      bool is_save = false, const string path = "");

  void generateHalfBoundedQueriesAndGroundtruthScalability(
      bool is_save = false, const string path = "");

  void generateHalfBoundedQueriesAndGroundtruthBenchmark(
      bool is_save_to_file, const string save_path = "");

  void generateRangeFilteringQueriesAndGroundtruthBenchmark(
      bool is_save_to_file, const string save_path = "");
};