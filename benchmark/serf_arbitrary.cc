/**
 * @file benchmark_arbitrary.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Benchmark Arbitrary Range Filter Search
 * @date 2024-11-17
 *
 * @copyright Copyright (c) 2024
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
// #include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "utils.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

void log_result_recorder(
    const std::map<int, std::pair<float, float>> &result_recorder,
    const std::map<int, float> &comparison_recorder, const int amount) {
  for (auto it : result_recorder) {
    cout << std::setiosflags(ios::fixed) << std::setprecision(4)
         << "range: " << it.first
         << "\t recall: " << it.second.first / (amount / result_recorder.size())
         << "\t QPS: " << std::setprecision(0)
         << (amount / result_recorder.size()) / it.second.second << "\t Comps: "
         << comparison_recorder.at(it.first) / (amount / result_recorder.size())
         << endl;
  }
}

int main(int argc, char **argv) {
#ifdef USE_SSE
  cout << "Use SSE" << endl;
#endif
#ifdef USE_AVX
  cout << "Use AVX" << endl;
#endif
#ifdef USE_AVX512
  cout << "Use AVX512" << endl;
#endif
#ifndef NO_PARALLEL_BUILD
  cout << "Index Construct Parallelly" << endl;
#endif

  // Parameters
  string dataset = "deep";
  int data_size = 100000;
  string dataset_path = "";
  string method = "";
  string query_path = "";
  string groundtruth_path = "";
  vector<int> index_k_list = {8};
  vector<int> ef_construction_list = {100};
  int query_num = 1000;
  int query_k = 10;
  vector<int> ef_max_list = {500};
  vector<int> searchef_para_range_list = {16, 64, 256};
  bool full_range = false;

  string indexk_str = "8";
  string ef_con_str = "100";
  string ef_max_str = "500";
  string ef_search_str = "16,64,256";
  string version = "Benchmark";

  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-dataset") dataset = string(argv[i + 1]);
    if (arg == "-N") data_size = atoi(argv[i + 1]);
    if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
    if (arg == "-query_path") query_path = string(argv[i + 1]);
    if (arg == "-groundtruth_path") groundtruth_path = string(argv[i + 1]);
    if (arg == "-index_k") indexk_str = string(argv[i + 1]);
    if (arg == "-ef_con") ef_con_str = string(argv[i + 1]);
    if (arg == "-ef_max") ef_max_str = string(argv[i + 1]);
    if (arg == "-ef_search") ef_search_str = string(argv[i + 1]);
    if (arg == "-method") method = string(argv[i + 1]);
    if (arg == "-full_range") full_range = true;
  }

  index_k_list = str2vec(indexk_str);
  ef_construction_list = str2vec(ef_con_str);
  ef_max_list = str2vec(ef_max_str);
  searchef_para_range_list = str2vec(ef_search_str);

  assert(index_k_list.size() != 0);
  assert(ef_construction_list.size() != 0);
  // assert(groundtruth_path != "");

  DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
  data_wrapper.readData(dataset_path, query_path);

  // Generate groundtruth
  if (full_range)
    data_wrapper.generateRangeFilteringQueriesAndGroundtruth(false);
  else
    data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(false);
  // Or you can load groundtruth from the given path
  // data_wrapper.LoadGroundtruth(groundtruth_path);

  assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

  cout << "index K:" << endl;
  print_set(index_k_list);
  cout << "ef construction:" << endl;
  print_set(ef_construction_list);
  cout << "search ef:" << endl;
  print_set(searchef_para_range_list);

  data_wrapper.version = version;

  base_hnsw::L2Space ss(data_wrapper.data_dim);

  timeval t1, t2;

  for (unsigned index_k : index_k_list) {
    for (unsigned ef_max : ef_max_list) {
      for (unsigned ef_construction : ef_construction_list) {
        BaseIndex::IndexParams i_params(index_k, ef_construction,
                                        ef_construction, ef_max);
        {
          cout << endl;
          i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
          SeRF::IndexSegmentGraph2D index(&ss, &data_wrapper);
          // rangeindex::RecursionIndex index(&ss, &data_wrapper);
          BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D",
                                            "benchmark");

          cout << "Method: " << search_info.method << endl;
          cout << "parameters: ef_construction ( " +
                      to_string(i_params.ef_construction) + " )  index-k( "
               << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
               << endl;
          gettimeofday(&t1, NULL);
          index.buildIndex(&i_params);
          gettimeofday(&t2, NULL);
          logTime(t1, t2, "Build Index Time");
          cout << "Total # of Neighbors: " << index.index_info->nodes_amount
               << endl;

          {
            timeval tt3, tt4;
            BaseIndex::SearchParams s_params;
            s_params.query_K = data_wrapper.query_k;
            for (auto one_searchef : searchef_para_range_list) {
              s_params.search_ef = one_searchef;
              std::map<int, std::pair<float, float>>
                  result_recorder;  // first->precision, second->query_time
              std::map<int, float> comparison_recorder;
              gettimeofday(&tt3, NULL);
              for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
                int one_id = data_wrapper.query_ids.at(idx);
                s_params.query_range =
                    data_wrapper.query_ranges.at(idx).second -
                    data_wrapper.query_ranges.at(idx).first + 1;
                auto res = index.rangeFilteringSearchOutBound(
                    &s_params, &search_info, data_wrapper.querys.at(one_id),
                    data_wrapper.query_ranges.at(idx));
                search_info.precision =
                    countPrecision(data_wrapper.groundtruth.at(idx), res);
                result_recorder[s_params.query_range].first +=
                    search_info.precision;
                result_recorder[s_params.query_range].second +=
                    search_info.internal_search_time;
                comparison_recorder[s_params.query_range] +=
                    search_info.total_comparison;
              }

              cout << endl
                   << "Search ef: " << one_searchef << endl
                   << "========================" << endl;
              log_result_recorder(result_recorder, comparison_recorder,
                                  data_wrapper.query_ids.size());
              cout << "========================" << endl;
              logTime(tt3, tt4, "total query time");
            }
          }
        }
      }
    }
  }

  return 0;
}
