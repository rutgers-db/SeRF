/**
 * Base Class of range filtering vector search index
 *
 * Author: Chaoji Zuo
 * Email:  chaoji.zuo@rutgers.edu
 * Date:   June 19, 2023
 */
#pragma once

#include <fstream>
#include <iostream>

#include "data_vecs.h"

namespace baseindex {

static unsigned const default_K = 16;
static unsigned const default_ef_construction = 400;

class BaseIndex {
 public:
  BaseIndex(hnswlib_incre::SpaceInterface<float>* s) {
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
  }
  // virtual void buildIndex() = 0;
  virtual ~BaseIndex() {}

  hnswlib_incre::DISTFUNC<float> fstdistfunc_;
  void* dist_func_param_;
  double sort_time;
  double nn_build_time;
  int num_search_comparison;

  // Indexing parameters
  struct IndexParams {
    // original params in hnsw
    unsigned K;
    unsigned ef_construction;
    unsigned random_seed;

    IndexParams()
        : K(default_K),
          ef_construction(default_ef_construction),
          random_seed(2023) {}
  };

  struct Searchparams {
    // vector<float> weights;
    // vector<std::pair<float, float>> weights;

    unsigned internal_search_K;
  };

  struct IndexInfo {};

  struct SearchInfo {
    SearchInfo(const DataWrapper* data, const IndexParams* index_params,
               const Searchparams* search_params, const string& meth,
               const string& ver) {
      data_wrapper = data;
      index = index_params;
      search = search_params;
      version = ver;
      method = meth;
      // save_path = "../exp/search/" + version + "-" + method + "-" +
      //             data_wrapper->dataset + "-" +
      //             std::to_string(data_wrapper->data_size) + ".csv";
      Path(ver);
    };
    const DataWrapper* data_wrapper;
    const IndexParams* index;
    const Searchparams* search;

    string method;
    string save_path;
    string version;

    double time;
    double precision;
    int query_id;
    unsigned break_counter;
    double internal_search_time;
    size_t visited_num;
    double fetch_nns_time;
    double cal_dist_time;
    double other_process_time;
    size_t total_comparison;
    size_t visited_num;
    size_t path_counter;
    
    bool is_investigate = false;

    void Path(const string& ver) {
      version = ver;
      save_path = "../exp/search/" + version + "-" + method + "-" +
                  data_wrapper->dataset + "-" +
                  std::to_string(data_wrapper->data_size) + ".csv";
      std::cout << "Save result to :" << save_path << std::endl;
    };

    void SaveCsv() {
      std::ofstream file;
      file.open(save_path, std::ios_base::app);
      if (file) {
        file <<
            // version << "," << method << "," <<
            time << "," << precision << "," << search->internal_search_K << ","
             << internal_search_time << "," << break_counter << ","
             << visited_num << "," << data_wrapper->data_size << "," << index->K
             << "," << index->ef_construction;
        file << "\n";
      }
      file.close();
    };

    void SavePathInvestigate(const float v1, const float v2, const float v3,
                             const float v4, bool is_new_row = false) {
      string investigate_path = "../exp/search/" + version + "-" + method +
                                "-" + data_wrapper->dataset + "-" +
                                std::to_string(data_wrapper->data_size) +
                                "-invetigate-path.csv";
      std::ofstream file;
      file.open(investigate_path, std::ios_base::app);

      if (is_new_row) {
        file << "\n";
      } else if (file) {
        file << v1 << "," << v2 << "," << v3 << "," << v4;
        file << "\n";
      }
      file.close();
    };
  };
};

}  // namespace baseindex
