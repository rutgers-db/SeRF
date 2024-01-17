/**
 * baseline #1, calculate nearest range query first, then search among the
 * range.
 *
 * Author: Chaoji Zuo
 * Date:   Nov 13, 2021
 * Email:  chaoji.zuo@rutgers.edu
 */
#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "incremental_hnsw/hnswlib.h"
#include "index_base.h"
#include "utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

void buildKNNFirstGraph(const vector<vector<float>> &nodes,
                        hnswlib_incre::HierarchicalNSW<float> &alg_hnsw) {
#pragma omp parallel for
  for (size_t i = 0; i < nodes.size(); ++i) {
    alg_hnsw.addPoint(nodes[i].data(), i);
  }
}

void addHNSWPointsSubgraph(const vector<vector<float>> &nodes,
                           hnswlib_incre::HierarchicalNSW<float> *alg_hnsw,
                           const int start, const int end) {
#pragma omp parallel for
  for (size_t i = start; i <= end; ++i) {
    alg_hnsw->addPoint(nodes[i].data(), i);
  }
}

void buildKNNFirstGraphSingleThread(
    const vector<vector<float>> &nodes,
    hnswlib_incre::HierarchicalNSW<float> &alg_hnsw) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    alg_hnsw.addPoint(nodes[i].data(), i);
  }
}

void buildKNNFirstGraphSingleThread(
    const vector<vector<float>> &nodes,
    hnswlib_incre::HierarchicalNSW<float> *alg_hnsw) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    alg_hnsw->addPoint(nodes[i].data(), i);
  }
}

vector<int> KNNFirstRangeSearch(
    const hnswlib_incre::HierarchicalNSW<float> &alg_hnsw,
    const vector<float> &query, const int l_bound, const int r_bound,
    const int query_k) {
  // K to run nndescent

  vector<int> result_in_range;
  auto res =
      alg_hnsw.searchKnnCloserFirst(query.data(), query_k, l_bound, r_bound);
  for (size_t j = 0; j < res.size(); j++) {
    int val = res[j].second;
    result_in_range.emplace_back(val);
  }
  return result_in_range;
}

vector<int> KNNFirstRangeSearchFixedEF(
    hnswlib_incre::HierarchicalNSW<float> &alg_hnsw, const vector<float> &query,
    const int l_bound, const int r_bound, const int query_k) {
  // K to run nndescent

  vector<int> result_in_range;
  auto res = alg_hnsw.searchKnnCloserFirst(query.data(), query_k, l_bound,
                                           r_bound, true);
  for (size_t j = 0; j < res.size(); j++) {
    int val = res[j].second;
    result_in_range.emplace_back(val);
  }

#ifdef LOG_DEBUG_MODE
  print_set(result_in_range);
  cout << l_bound << "," << r_bound << endl;
  assert(false);
#endif

  return result_in_range;
}

vector<int> KNNFirstRangeSearchFixedEF(
    hnswlib_incre::HierarchicalNSW<float> *alg_hnsw, const vector<float> &query,
    const int l_bound, const int r_bound, const int query_k,
    const int search_ef) {
  // K to run nndescent

  alg_hnsw->setEf(search_ef);

  vector<int> result_in_range;
  auto res = alg_hnsw->searchKnnCloserFirst(query.data(), query_k, l_bound,
                                            r_bound, true);
  for (size_t j = 0; j < res.size(); j++) {
    int val = res[j].second;
    result_in_range.emplace_back(val);
  }
  return result_in_range;
}

class KnnFirstWrapper : BaseIndex {
 public:
  KnnFirstWrapper(const DataWrapper *data) : BaseIndex(data) {
    index_info = new IndexInfo();
    index_info->index_version_type = "KnnFirst-hnsw";
  };

  IndexInfo *index_info;

  hnswlib_incre::HierarchicalNSW<float> *hnsw_index;
  hnswlib_incre::L2Space *space;

  void countNeighbrs() {
    int node_amount = 0;

    for (unsigned idx = 0; idx < data_wrapper->data_size; idx++) {
      hnswlib_incre::linklistsizeint *linklist;
      linklist = hnsw_index->get_linklist0(idx);
      size_t linklist_count = hnsw_index->getListCount(linklist);
      node_amount += linklist_count;
    }
    index_info->nodes_amount = node_amount;
    index_info->avg_forward_nns = (float)node_amount / data_wrapper->data_size;
    cout << "# of Avg. Neighbors: " << index_info->avg_forward_nns << endl;
  }

  void buildIndex(const IndexParams *index_params) override {
    cout << "Building baseline graph: " << index_info->index_version_type
         << endl;

    timeval tt1, tt2;
    gettimeofday(&tt1, NULL);
    space = new hnswlib_incre::L2Space(data_wrapper->data_dim);

    hnsw_index = new hnswlib_incre::HierarchicalNSW<float>(
        space, 2 * data_wrapper->data_size, index_params->K,
        index_params->ef_construction);
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
      hnsw_index->addPoint(data_wrapper->nodes.at(i).data(), i);
    }
    cout << "Done" << endl;
    index_info->index_time = CountTime(tt1, tt2);
    countNeighbrs();
  }

  vector<int> rangeFilteringSearchInRange(
      const SearchParams *search_params, SearchInfo *search_info,
      const vector<float> &query,
      const std::pair<int, int> query_bound) override {
    return rangeFilteringSearchOutBound(search_params, search_info, query,
                                        query_bound);
  }

  vector<int> rangeFilteringSearchOutBound(
      const SearchParams *search_params, SearchInfo *search_info,
      const vector<float> &query,
      const std::pair<int, int> query_bound) override {
    timeval tt1, tt2;

    hnsw_index->search_info = search_info;
    gettimeofday(&tt1, NULL);

    hnsw_index->setEf(search_params->search_ef);

    vector<int> result_in_range;
    auto res = hnsw_index->searchKnnCloserFirst(
        query.data(), search_params->query_K, query_bound.first,
        query_bound.second, true);
    for (size_t j = 0; j < res.size(); j++) {
      int val = res[j].second;
      result_in_range.emplace_back(val);
    }
    gettimeofday(&tt2, NULL);
    CountTime(tt1, tt2, search_info->internal_search_time);
    return result_in_range;
  }

  void saveIndex(const string &save_path) { hnsw_index->saveIndex(save_path); }

  ~KnnFirstWrapper() {
    delete hnsw_index;
    delete index_info;
    delete space;
  }
};

void execute_knn_first_search(KnnFirstWrapper &index,
                              BaseIndex::SearchInfo &search_info,
                              const DataWrapper &data_wrapper,
                              const vector<int> &searchef_para_range_list) {
  timeval tt3, tt4;
  for (auto one_searchef : searchef_para_range_list) {
    gettimeofday(&tt3, NULL);
    for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
      int one_id = data_wrapper.query_ids.at(idx);
      BaseIndex::SearchParams s_params;
      s_params.query_K = data_wrapper.query_k;
      s_params.search_ef = one_searchef;
      s_params.control_batch_threshold = 1;
      s_params.query_range = data_wrapper.query_ranges.at(idx).second -
                             data_wrapper.query_ranges.at(idx).first;
      auto res = index.rangeFilteringSearchOutBound(
          &s_params, &search_info, data_wrapper.querys.at(one_id),
          data_wrapper.query_ranges.at(idx));
      search_info.precision =
          countPrecision(data_wrapper.groundtruth.at(idx), res);
      search_info.approximate_ratio = countApproximationRatio(
          data_wrapper.nodes, data_wrapper.groundtruth.at(idx), res,
          data_wrapper.querys.at(one_id));

      // cout << data_wrapper.query_ranges.at(idx).first << "  "
      //      << data_wrapper.query_ranges.at(idx).second << endl;
      // print_set(res);
      // print_set(data_wrapper.groundtruth.at(idx));
      // cout << endl;

      search_info.RecordOneQuery(&s_params);
    }

    logTime(tt3, tt4, "total query time");
  }
}

void execute_knn_first_search_groundtruth_wrapper(
    KnnFirstWrapper &index, BaseIndex::SearchInfo &search_info,
    const DataWrapper &data_wrapper, const DataWrapper &groundtruth_wrapper,
    const vector<int> &searchef_para_range_list) {
  timeval tt3, tt4;
  for (auto one_searchef : searchef_para_range_list) {
    gettimeofday(&tt3, NULL);
    for (int idx = 0; idx < groundtruth_wrapper.query_ids.size(); idx++) {
      int one_id = groundtruth_wrapper.query_ids.at(idx);
      BaseIndex::SearchParams s_params;
      s_params.query_K = data_wrapper.query_k;
      s_params.search_ef = one_searchef;
      s_params.control_batch_threshold = 1;
      s_params.query_range = groundtruth_wrapper.query_ranges.at(idx).second -
                             groundtruth_wrapper.query_ranges.at(idx).first;
      auto res = index.rangeFilteringSearchOutBound(
          &s_params, &search_info, data_wrapper.querys.at(one_id),
          groundtruth_wrapper.query_ranges.at(idx));
      search_info.precision =
          countPrecision(groundtruth_wrapper.groundtruth.at(idx), res);
      search_info.approximate_ratio = countApproximationRatio(
          data_wrapper.nodes, groundtruth_wrapper.groundtruth.at(idx), res,
          data_wrapper.querys.at(one_id));

      search_info.RecordOneQuery(&s_params);
    }

    logTime(tt3, tt4, "total query time");
  }
}