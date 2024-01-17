/**
 * @file index_base.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Base class for builiding segment graph index, containing virtual
 * function about indexing and searching.
 * @date Revised: 2024-01-11
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>

#include "base_hnsw/space_l2.h"
#include "data_wrapper.h"
#include "utils.h"

using std::cout;
using std::endl;
using std::vector;

// TODO: manage default parameter in the same place
static unsigned const default_K = 16;
static unsigned const default_ef_construction = 400;

class BaseIndex {
 public:
  BaseIndex(const DataWrapper* data) { data_wrapper = data; }

  int num_search_comparison;
  int k_graph_out_bound;
  bool isLog = true;

  // Indexing parameters
  struct IndexParams {
    // original params in hnsw
    unsigned K;  // out degree boundry
    unsigned ef_construction = 400;
    unsigned random_seed = 100;
    unsigned ef_large_for_pruning = 400;  // TODO: Depratched parameter
    unsigned ef_max = 2000;
    unsigned ef_construction_2d_max; // Replace ef_max
    bool print_one_batch = false;

    IndexParams(unsigned K, unsigned ef_construction,
                unsigned ef_large_for_pruning, unsigned ef_max)
        : K(K),
          ef_construction(ef_construction),
          ef_large_for_pruning(ef_large_for_pruning),
          ef_max(ef_max){};

    // which position to cut during the recursion
    enum Recursion_Type_t { MIN_POS, MID_POS, MAX_POS, SMALL_LEFT_POS };
    Recursion_Type_t recursion_type = Recursion_Type_t::MAX_POS;
    IndexParams()
        : K(default_K),
          ef_construction(default_ef_construction),
          random_seed(2023) {}
  };

  struct IndexInfo {
    string index_version_type;
    double index_time;
    int window_count;
    int nodes_amount;
    float avg_forward_nns;
    float avg_reverse_nns;
  };

  struct SearchParams {
    unsigned query_K;
    unsigned search_ef;
    unsigned query_range;
    float control_batch_threshold = 1;
  };

  struct SearchInfo {
    SearchInfo(const DataWrapper* data,
               const BaseIndex::IndexParams* index_params, const string& meth,
               const string& ver) {
      data_wrapper = data;
      index = index_params;
      version = ver;
      method = meth;
      path_counter = 0;
      Path(ver + "-" + data->version);
    };

    const DataWrapper* data_wrapper;
    const BaseIndex::IndexParams* index;
    string version;
    string method;

    int index_k;

    double time;
    double precision;
    double approximate_ratio;
    int query_id;
    double internal_search_time;  // one query time
    double fetch_nns_time = 0;
    double cal_dist_time = 0;
    double other_process_time = 0;
    // double one_query_time;
    size_t total_comparison = 0;
    // size_t visited_num;
    size_t path_counter;
    string investigate_path;
    string save_path;

    bool is_investigate = false;

    void Path(const string& ver) {
      version = ver;
      save_path = "../exp/search/" + version + "-" + method + "-" +
                  data_wrapper->dataset + "-" +
                  std::to_string(data_wrapper->data_size) + ".csv";

      std::cout << "Save result to :" << save_path << std::endl;
    };

    void RecordOneQuery(BaseIndex::SearchParams* search) {
      std::ofstream file;
      file.open(save_path, std::ios_base::app);
      if (file) {
        file <<
            // version << "," << method << "," <<
            internal_search_time << "," << precision << "," << approximate_ratio
             << "," << search->query_range << "," << search->search_ef << ","
             << fetch_nns_time << "," << cal_dist_time << ","
             << total_comparison << "," << std::to_string(index->recursion_type)
             << "," << index->K << "," << index->ef_max << ","
             << index->ef_large_for_pruning << "," << index->ef_construction;
        file << "\n";
      }
      file.close();
    }
  };

  const DataWrapper* data_wrapper;
  SearchInfo* search_info;

  virtual void buildIndex(const IndexParams* index_params) = 0;
  virtual vector<int> rangeFilteringSearchInRange(
      const SearchParams* search_params, SearchInfo* search_info,
      const vector<float>& query, const std::pair<int, int> query_bound) = 0;
  virtual vector<int> rangeFilteringSearchOutBound(
      const SearchParams* search_params, SearchInfo* search_info,
      const vector<float>& query, const std::pair<int, int> query_bound) = 0;
  virtual ~BaseIndex() {}
};
