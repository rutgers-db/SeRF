/**
 * @file faiss_ivfpq_halfbound.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Use faiss selector to query range filter anns
 * @date 2023-07-03
 *
 * @copyright Copyright (c) 2023
 */

#include <faiss/IVFlib.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <sys/time.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// #include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "logger.h"
#include "range_index_base.h"
#include "reader.h"
#include "utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int query_num = 500;

int exp(string dataset, int data_size, string dataset_path, string method,
        string query_path, const bool is_amarel, const string gt_path,
        const string version) {
#ifdef USE_SSE
  cout << "Use SSE" << endl;
#endif

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

  data_wrapper.LoadGroundtruth(gt_path);

  data_wrapper.version = version;
  if (is_amarel) {
    data_wrapper.version = "amarel-" + version;
  }
  vector<int> nlist_list = {128, 256, 512};
  //   vector<int> nlist_list = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

  vector<int> nprobe_list = {50, 100, 200, 400, 800};

  // size of the database we plan to index
  size_t nb = data_size;

  faiss::IndexFlatL2 coarse_quantizer(
      data_wrapper.data_dim);  // the other index
  faiss::MetricType metric = faiss::METRIC_L2;

  std::vector<float> database(nb * data_wrapper.data_dim);
  for (size_t i = 0; i < nb * data_wrapper.data_dim; i++) {
    database[i] = data_wrapper.nodes.at(i / data_wrapper.data_dim)
                      .at(i % data_wrapper.data_dim);
  }

  timeval t1, t2;

  vector<int> M_divide_list = {2, 4};
  print_set(data_wrapper.groundtruth.at(0));

  for (auto nlist : nlist_list) {
    // int nlist = 32;
    for (auto M_divide : M_divide_list) {
      BaseIndex::IndexParams i_params;
      i_params.ef_large_for_pruning = 0;
      i_params.ef_max = 0;
      i_params.ef_construction = data_wrapper.data_dim / M_divide;
      i_params.K = nlist;
      BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "Faiss-IVFPQ",
                                        "twoside/search");
      cout << endl
           << "nlist: " << nlist << " M: " << data_wrapper.data_dim / M_divide
           << endl;

      gettimeofday(&t1, NULL);

      faiss::IndexIVFPQ index(&coarse_quantizer, data_wrapper.data_dim, nlist,
                              data_wrapper.data_dim / M_divide, 8);
      // faiss::IndexRefineFlat index(&pq_index);

      index.verbose = true;
      index.train(nb, database.data());
      logTime(t1, t2, "Training the index");

      size_t nq = data_wrapper.query_ids.size();

      {  // populating the database
        gettimeofday(&t1, NULL);
        index.add(nb, database.data());
        logTime(t1, t2, "Building the index");
      }

      {  // searching one by one
        int k = data_wrapper.query_k;

        for (size_t idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
          auto one_id = data_wrapper.query_ids.at(idx);

          std::vector<faiss::idx_t> nns(k);
          std::vector<float> dis(k);

          faiss::IDSelectorRange ivf_search_range(
              data_wrapper.query_ranges.at(idx).first,
              data_wrapper.query_ranges.at(idx).second + 1, true);

          for (auto nprobe : nprobe_list) {
            faiss::SearchParametersIVF search_params;
            search_params.sel = &ivf_search_range;
            search_params.nprobe = nprobe;
            timeval temp1, temp2;
            gettimeofday(&temp1, NULL);

            index.search(1, data_wrapper.querys.at(one_id).data(), k,
                         dis.data(), nns.data(), &search_params);

            vector<int> pred;
            pred.insert(pred.end(), nns.begin(), nns.begin() + k);
            gettimeofday(&temp2, NULL);
            CountTime(temp1, temp2, search_info.internal_search_time);

            search_info.precision =
                countPrecision(data_wrapper.groundtruth.at(idx), pred);
            // print_set(data_wrapper.groundtruth.at(idx));
            // print_set(pred);

            search_info.approximate_ratio = countApproximationRatio(
                data_wrapper.nodes, data_wrapper.groundtruth.at(idx), pred,
                data_wrapper.querys.at(one_id));

            BaseIndex::SearchParams s_params;
            s_params.search_ef = nprobe;
            s_params.query_K = data_wrapper.query_k;
            s_params.query_range = data_wrapper.query_ranges.at(idx).second -
                                   data_wrapper.query_ranges.at(idx).first;

            search_info.RecordOneQuery(&s_params);
          }
        }
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  string dataset = "biggraph";
  int data_size = 100000;
  string dataset_path = "";
  string method = "";
  string query_path = "";
  string groundtruth_path = "";
  bool is_amarel = false;
  string version = "";

  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-dataset") dataset = string(argv[i + 1]);
    if (arg == "-N") data_size = atoi(argv[i + 1]);
    if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
    if (arg == "-method") method = string(argv[i + 1]);
    if (arg == "-query_path") query_path = string(argv[i + 1]);
    if (arg == "-amarel") is_amarel = true;
    if (arg == "-groundtruth_path") groundtruth_path = string(argv[i + 1]);
    if (arg == "-version") version = string(argv[i + 1]);
  }
  assert(groundtruth_path != "");

  exp(dataset, data_size, dataset_path, method, query_path, is_amarel,
      groundtruth_path, version);
  return 0;
}