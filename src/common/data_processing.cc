/**
 * @file data_processing.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Functions for processing data, generating querys and groundtruth
 * @date 2023-06-19
 *
 * @copyright Copyright (c) 2023
 */

#include "data_processing.h"

#include "data_wrapper.h"
#include "omp.h"

using std::pair;

// void SynthesizeQuerys(const vector<vector<float>> &nodes,
//                       vector<vector<float>> &querys, const int query_num) {
//   int dim = nodes.front().size();
//   std::default_random_engine e;
//   std::uniform_int_distribution<int> u(0, nodes.size() - 1);
//   querys.clear();
//   querys.resize(query_num);

//   for (unsigned n = 0; n < query_num; n++) {
//     for (unsigned i = 0; i < dim; i++) {
//       int select_idx = u(e);
//       querys[n].emplace_back(nodes[select_idx][i]);
//     }
//   }
// }

// vector<int> greedyNearest(const vector<vector<float>> &dpts,
//                           const vector<float> query, const int k_smallest) {
//   std::priority_queue<std::pair<float, int>> top_candidates;
//   float lower_bound = _INT_MAX;
//   for (size_t i = 0; i < dpts.size(); i++) {
//     float dist = EuclideanDistance(query, dpts[i]);
//     if (top_candidates.size() < k_smallest || dist < lower_bound) {
//       top_candidates.push(std::make_pair(dist, i));
//       if (top_candidates.size() > k_smallest) {
//         top_candidates.pop();
//       }

//       lower_bound = top_candidates.top().first;
//     }
//   }
//   vector<int> res;
//   while (!top_candidates.empty()) {
//     res.emplace_back(top_candidates.top().second);
//     top_candidates.pop();
//   }
//   std::reverse(res.begin(), res.end());
//   return res;
// }

// // generate range filtering querys and calculate groundtruth
// void calculateGroundtruth(DataWrapper &data_wrapper) {
//   std::default_random_engine e;
//   vector<pair<int, int>> query_ranges;
//   vector<vector<int>> groundtruth;
//   vector<int> query_ids;
//   timeval t1, t2;
//   double accu_time = 0.0;

//   vector<int> query_range_list;
//   float scale = 0.05;
//   if (data_wrapper.dataset == "local") {
//     scale = 0.1;
//   }
//   int init_range = data_wrapper.data_size * scale;
//   while (init_range < data_wrapper.data_size) {
//     query_range_list.emplace_back(init_range);
//     init_range += data_wrapper.data_size * scale;
//   }
// #ifdef LOG_DEBUG_MODE
//   if (data_wrapper.dataset == "local") {
//     query_range_list.erase(
//         query_range_list.begin(),
//         query_range_list.begin() + 4 * query_range_list.size() / 9);
//   }
// #endif

//   cout << "Generating Groundtruth...";
//   // generate groundtruth

//   for (auto range : query_range_list) {
//     if (range > runner.data_size - 100) {
//       break;
//     }
//     uniform_int_distribution<int> u_lbound(0,
//                                            data_wrapper.data_size - range - 80);
//     for (int i = 0; i < data_wrapper.querys.size(); i++) {
//       int l_bound = u_lbound(e);
//       int r_bound = l_bound + range;
//       int search_key_range = r_bound - l_bound;
//       if (data_wrapper.real_keys)
//         search_key_range = data_wrapper.nodes_keys.at(r_bound) -
//                            data_wrapper.nodes_keys.at(l_bound);
//       for (auto query_K : query_k_range_list) {
//         query_ranges.emplace_back(make_pair(l_bound, r_bound));
//         double greedy_time;
//         gettimeofday(&t1, NULL);
//         auto gt = greedyNearest(data_wrapper.nodes, runner.querys.at(i),
//                                 l_bound, r_bound, query_K);
//         gettimeofday(&t2, NULL);
//         CountTime(t1, t2, greedy_time);
//         // SaveToCSVRow("../exp_result/exp2-amarel-scalability-greedy-baseline-"
//         // +
//         //                  to_string(query_K) + "-" + to_string(nodes.size()) +
//         //                  "-" + dataset + ".csv",
//         //              i, l_bound, r_bound, r_bound - l_bound,
//         //              search_key_range, query_K, greedy_time, gt);
//         groundtruth.emplace_back(gt);
//         query_ids.emplace_back(i);
//       }
//     }
//   }

//   cout << " Done!" << endl << "Groundtruth Time: " << accu_time << endl;

//   data_wrapper.groundtruth = std::move(groundtruth);
//   data_wrapper.query_ids = std::move(query_ids);
//   data_wrapper.query_ranges = std::move(query_ranges);
// }

// // Get Groundtruth for half bounded query
// void calculateGroundtruthHalfBounded(ExpRunner &runner, bool is_save = false) {
//   default_random_engine e;
//   vector<pair<int, int>> query_ranges;
//   vector<vector<int>> groundtruth;
//   vector<int> query_ids;

//   timeval t1, t2;

//   vector<int> query_k_range_list = {10};
//   vector<int> query_range_list;
//   int init_range = runner.data_size * 0.05;
//   while (init_range < runner.data_size) {
//     query_range_list.emplace_back(init_range);
//     init_range += runner.data_size * 0.05;
//   }

//   cout << "Generating Groundtruth...";
//   // generate groundtruth
//   for (auto range : query_range_list) {
//     if (range > runner.data_size - 100) {
//       break;
//     }
//     for (int i = 0; i < runner.querys.size(); i++) {
//       // int l_bound = range - 1;
//       // int r_bound = runner.data_size - 1;

//       int l_bound = 0;
//       int r_bound = range;

//       int search_key_range = r_bound - l_bound;
//       if (runner.real_keys)
//         search_key_range =
//             runner.nodes_keys.at(r_bound) - runner.nodes_keys.at(l_bound);
//       for (auto query_K : query_k_range_list) {
//         query_ranges.emplace_back(make_pair(l_bound, r_bound));
//         double greedy_time;
//         gettimeofday(&t1, NULL);
//         auto gt = greedyNearest(runner.nodes, runner.querys.at(i), l_bound,
//                                 r_bound, query_K);
//         gettimeofday(&t2, NULL);
//         CountTime(t1, t2, greedy_time);
//         // SaveToCSVRow("../exp_result/exp2-amarel-scalability-greedy-baseline-"
//         // +
//         //                  to_string(query_K) + "-" + to_string(nodes.size()) +
//         //                  "-" + dataset + ".csv",
//         //              i, l_bound, r_bound, r_bound - l_bound,
//         //              search_key_range, query_K, greedy_time, gt);
//         groundtruth.emplace_back(gt);
//         query_ids.emplace_back(i);
//       }
//     }
//   }

//   runner.query_ids.swap(query_ids);
//   runner.query_ranges.swap(query_ranges);
//   runner.groundtruth.swap(groundtruth);
//   cout << "  Done!" << endl;
// }