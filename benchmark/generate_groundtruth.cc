/**
 * @file exp_halfbound.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Benchmark Half-Bounded Range Filter Search
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "data_processing.h"
#include "data_wrapper.h"
#include "logger.h"
#include "index_base.h"
#include "reader.h"
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

int main(int argc, char **argv) {
  // Parameters
  string dataset = "deep";
  int data_size = 100000;
  string dataset_path = "";
  string query_path = "";
  string groundtruth_prefix = "";
  int query_num = 1000;
  int query_k = 10;

  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    // if (arg == "-dataset") dataset = string(argv[i + 1]);
    if (arg == "-N") data_size = atoi(argv[i + 1]);
    if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
    if (arg == "-query_path") query_path = string(argv[i + 1]);
    if (arg == "-groundtruth_prefix") groundtruth_prefix = string(argv[i + 1]);
  }

  string size_symbol = "";
  if (data_size == 100000) {
    size_symbol = "100k";
  } else if (data_size == 1000000) {
    size_symbol = "1m";
  }

  DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
  data_wrapper.readData(dataset_path, query_path);

  // data_wrapper.generateHalfBoundedQueriesAndGroundtruth(
  //     true, groundtruth_prefix + "benchmark-groundtruth-deep-" + size_symbol
  //     +
  //               "-num1000-k10.halfbounded.cvs");
  // data_wrapper.generateRangeFilteringQueriesAndGroundtruth(
  //     true, groundtruth_prefix + "benchmark-groundtruth-deep-" + size_symbol
  //     +
  //               "-num1000-k10.arbitrary.cvs");

  data_wrapper.generateHalfBoundedQueriesAndGroundtruthBenchmark(
      true, groundtruth_prefix + "benchmark-groundtruth-deep-" + size_symbol +
                "-num1000-k10.halfbounded.cvs");
  data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(
      true, groundtruth_prefix + "benchmark-groundtruth-deep-" + size_symbol +
                "-num1000-k10.arbitrary.cvs");
  return 0;
}