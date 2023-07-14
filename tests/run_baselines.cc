/**
 * @file run_baselines.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief RunBaselines
 * @date 2023-06-20
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "logger.h"
#include "range_index_base.h"
#include "reader.h"
#include "utils.h"

// #define LOG_DEBUG_MODE 1

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

long long before_memory, after_memory;

int exp(string dataset, int data_size, string dataset_path, string method,
        string query_path, const bool is_amarel, const string gt_path,
        vector<int> index_k_list, vector<int> ef_construction_list,
        string version) {
#ifdef USE_SSE
  cout << "Use SSE" << endl;
#endif

  cout << "method: " << method << endl;
  int query_num = 500;
  vector<vector<float>> nodes;
  vector<int> search_keys;
  vector<vector<float>> querys;
  vector<int> querys_keys;

  if (dataset == "local") {
    cout << "Local test" << endl;
    query_num = 10;
  }

  int query_k = 10;
  DataWrapper data_wrapper(query_num, query_k, dataset, data_size);

  data_wrapper.readData(dataset_path, query_path);

  vector<int> searchef_para_range_list = {50, 100, 200, 300, 400, 800};

  version = "baseline-" + version;
  if (is_amarel) {
    version = "amarel-" + version;
  }

  data_wrapper.LoadGroundtruth(gt_path);

  assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

  delta_index_hnsw_full_reverse::L2Space ss(data_wrapper.data_dim);

  timeval t1, t2, t3, t4;

  data_wrapper.version = version;
  for (unsigned ef_constrcution : ef_construction_list) {
    for (unsigned index_k : index_k_list) {
      BaseIndex::IndexParams i_params;
      i_params.ef_construction = ef_constrcution;
      i_params.K = index_k;

      i_params.ef_large_for_pruning = 0;
      i_params.ef_max = 0;
      {
        // Baseline
        cout << endl;
        BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
                                          "KnnFirstHnsw", "twoside/search");
        cout << "Method: " << search_info.method << endl;
        KnnFirstWrapper index(&data_wrapper);
        gettimeofday(&t1, NULL);
        index.buildIndex(&i_params);
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Build knnfirst HNSW Index Time");
        cout << "HNSW Total # of Neighbors: " << index.index_info->nodes_amount
             << endl;
        execute_knn_first_search(index, search_info, data_wrapper,
                                 searchef_para_range_list);
      }
    }
  }

  return 0;
}

vector<int> str2vec(const string str) {
  std::vector<int> vect;
  std::stringstream ss(str);
  for (int i; ss >> i;) {
    vect.push_back(i);
    if (ss.peek() == ',') ss.ignore();
  }
  return vect;
}

int main(int argc, char **argv) {
  string dataset = "biggraph";
  int data_size = 100000;
  string dataset_path = "";
  string method = "";
  string query_path = "";
  bool is_amarel = false;
  string groundtruth_path = "";
  vector<int> index_k_list;
  vector<int> ef_construction_list;
  vector<int> ef_max_list;
  string indexk_str = "";
  string ef_con_str = "";
  string ef_max_str = "";
  string version = "knnfirst";
  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-dataset") dataset = string(argv[i + 1]);
    if (arg == "-N") data_size = atoi(argv[i + 1]);
    if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
    if (arg == "-method") method = string(argv[i + 1]);
    if (arg == "-query_path") query_path = string(argv[i + 1]);
    if (arg == "-amarel") is_amarel = true;
    if (arg == "-groundtruth_path") groundtruth_path = string(argv[i + 1]);
    if (arg == "-index_k") indexk_str = string(argv[i + 1]);
    if (arg == "-ef_construction") ef_con_str = string(argv[i + 1]);
    if (arg == "-version") version = string(argv[i + 1]);
  }

  index_k_list = str2vec(indexk_str);
  ef_construction_list = str2vec(ef_con_str);
  ef_max_list = str2vec(ef_max_str);

  cout << "index K:" << endl;
  print_set(index_k_list);
  cout << "ef construction:" << endl;
  print_set(ef_construction_list);

  assert(index_k_list.size() != 0);
  assert(ef_construction_list.size() != 0);
  cout << "Running: " << version << endl;

  exp(dataset, data_size, dataset_path, method, query_path, is_amarel,
      groundtruth_path, index_k_list, ef_construction_list, version);
  return 0;
}