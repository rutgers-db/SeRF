#include "data_wrapper.h"

#include "reader.h"
#include "utils.h"

void SynthesizeQuerys(const vector<vector<float>> &nodes,
                      vector<vector<float>> &querys, const int query_num) {
  int dim = nodes.front().size();
  std::default_random_engine e;
  std::uniform_int_distribution<int> u(0, nodes.size() - 1);
  querys.clear();
  querys.resize(query_num);

  for (unsigned n = 0; n < query_num; n++) {
    for (unsigned i = 0; i < dim; i++) {
      int select_idx = u(e);
      querys[n].emplace_back(nodes[select_idx][i]);
    }
  }
}

void DataWrapper::readData(string &dataset_path, string &query_path) {
  ReadDataWrapper(dataset, dataset_path, this->nodes, data_size, query_path,
                  this->querys, query_num, this->nodes_keys);
  cout << "Load vecs from: " << dataset_path << endl;
  cout << "# of vecs: " << nodes.size() << endl;

  // already sort data in sorted_data
  // if (dataset != "wiki-image" && dataset != "yt8m") {
  //   nodes_keys.resize(nodes.size());
  //   iota(nodes_keys.begin(), nodes_keys.end(), 0);
  // }

  if (querys.empty()) {
    cout << "Synthesizing querys..." << endl;
    SynthesizeQuerys(nodes, querys, query_num);
  }

  this->real_keys = false;
  vector<size_t> index_permutation;  // already sort data ahead

  // if (dataset == "wiki-image" || dataset == "yt8m") {
  //   cout << "first search_key before sorting: " << nodes_keys.front() <<
  //   endl; cout << "sorting dataset: " << dataset << endl; index_permutation =
  //   sort_permutation(nodes_keys); apply_permutation_in_place(nodes,
  //   index_permutation); apply_permutation_in_place(nodes_keys,
  //   index_permutation); cout << "Dimension: " << nodes.front().size() <<
  //   endl; cout << "first search_key: " << nodes_keys.front() << endl;
  //   this->real_keys = true;
  // }
  this->data_dim = this->nodes.front().size();
}

void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int pos_range,
                  const int real_search_key_range, const int K_neighbor,
                  const double &search_time, const vector<int> &gt,
                  const vector<float> &pts) {
  std::ofstream file;
  file.open(path, std::ios_base::app);
  if (file) {
    file << idx << "," << l_bound << "," << r_bound << "," << pos_range << ","
         << real_search_key_range << "," << K_neighbor << "," << search_time
         << ",";
    for (auto ele : gt) {
      file << ele << " ";
    }
    // file << ",";
    // for (auto ele : pts) {
    //   file << ele << " ";
    // }
    file << "\n";
  }
  file.close();
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruth(
    bool is_save_to_file, const string save_path) {
  std::default_random_engine e;
  timeval t1, t2;
  double accu_time = 0.0;

  vector<int> query_range_list;
  float scale = 0.05;
  if (this->dataset == "local") {
    scale = 0.1;
  }
  int init_range = this->data_size * scale;
  while (init_range <= this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }
#ifdef LOG_DEBUG_MODE
  if (this->dataset == "local") {
    query_range_list.erase(
        query_range_list.begin(),
        query_range_list.begin() + 4 * query_range_list.size() / 9);
  }
#endif
  if (0.01 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.01 * this->data_size);

  if (0.001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.001 * this->data_size);

  if (0.0001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.0001 * this->data_size);
  if (this->data_size == 1000000) {
    query_range_list = {1000,   2000,   3000,   4000,   5000,   6000,
                        7000,   8000,   9000,   10000,  20000,  30000,
                        40000,  50000,  60000,  70000,  80000,  90000,
                        100000, 200000, 300000, 400000, 500000, 600000,
                        700000, 800000, 900000, 1000000};
  }

  cout << "Generating Groundtruth...\nRanges: ";
  print_set(query_range_list);
  // generate groundtruth

  for (auto range : query_range_list) {
    std::uniform_int_distribution<int> u_lbound(0,
                                                this->data_size - range - 80);
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = u_lbound(e);
      int r_bound = l_bound + range - 1;
      if (range == this->data_size) {
        l_bound = 0;
        r_bound = this->data_size - 1;
      }
      int search_key_range = r_bound - l_bound + 1;
      if (this->real_keys)
        search_key_range =
            this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      accu_time += greedy_time;
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }

  cout << " Done!" << endl << "Groundtruth Time: " << accu_time << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

// Get Groundtruth for half bounded query
void DataWrapper::generateHalfBoundedQueriesAndGroundtruth(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  vector<int> query_range_list;
  int init_range = this->data_size * 0.05;
  while (init_range <= this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * 0.05;
  }
  if (0.01 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.01 * this->data_size);

  if (0.001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.001 * this->data_size);

  if (0.0001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.0001 * this->data_size);

  if (this->data_size == 1000000) {
    query_range_list = {1000,   2000,   3000,   4000,   5000,   6000,
                        7000,   8000,   9000,   10000,  20000,  30000,
                        40000,  50000,  60000,  70000,  80000,  90000,
                        100000, 200000, 300000, 400000, 500000, 600000,
                        700000, 800000, 900000, 1000000};
  }

  if (this->data_size == 100000) {
    query_range_list = {1000,  2000,  3000,  4000,  5000,  6000,  7000,
                        8000,  9000,  10000, 20000, 30000, 40000, 50000,
                        60000, 70000, 80000, 90000, 100000};
  }

  cout << "Generating Half Bounded Groundtruth...";
  cout << endl << "Ranges: " << endl;
  print_set(query_range_list);
  // generate groundtruth
  for (auto range : query_range_list) {
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = 0;
      int r_bound = range - 1;

      int search_key_range = r_bound - l_bound + 1;
      if (this->real_keys)
        search_key_range =
            this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }
  cout << "  Done!" << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

void DataWrapper::LoadGroundtruth(const string &gt_path) {
  this->groundtruth.clear();
  this->query_ranges.clear();
  this->query_ids.clear();
  cout << "Loading Groundtruth from" << gt_path << "...";
  ReadGroundtruthQuery(this->groundtruth, this->query_ranges, this->query_ids,
                       gt_path);
  cout << "    Done!" << endl;
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruthScalability(
    bool is_save_to_file, const string save_path) {
  std::default_random_engine e;
  timeval t1, t2;
  double accu_time = 0.0;

  vector<int> query_range_list;
  float scale = 0.001;
  int init_range = this->data_size * scale;
  while (init_range < 0.01 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }
  scale = 0.01;
  init_range = this->data_size * scale;
  while (init_range < 0.1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  scale = 0.1;
  init_range = this->data_size * scale;
  while (init_range < 1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  query_range_list.emplace_back(this->data_size);

  cout << "Generating Groundtruth...\nRanges: ";
  print_set(query_range_list);
  cout << "sample size:" << this->nodes.size() << endl;
  // generate groundtruth

  for (auto range : query_range_list) {
    std::uniform_int_distribution<int> u_lbound(0,
                                                this->data_size - range - 80);
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = u_lbound(e);
      int r_bound = l_bound + range - 1;
      if (range == this->data_size) {
        l_bound = 0;
        r_bound = this->data_size - 1;
      }
      int search_key_range = r_bound - l_bound + 1;
      // if (this->real_keys)
      //   search_key_range =
      //       this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      accu_time += greedy_time;
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }

  cout << " Done!" << endl << "Groundtruth Time: " << accu_time << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

// Get Groundtruth for half bounded query
void DataWrapper::generateHalfBoundedQueriesAndGroundtruthScalability(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  vector<int> query_range_list;
  float scale = 0.001;
  int init_range = this->data_size * scale;
  while (init_range < 0.01 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }
  scale = 0.01;
  init_range = this->data_size * scale;
  while (init_range < 0.1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  scale = 0.1;
  init_range = this->data_size * scale;
  while (init_range < 1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  query_range_list.emplace_back(this->data_size);

  cout << "Generating Half Bounded Groundtruth...";
  cout << endl << "Ranges: " << endl;
  print_set(query_range_list);
  // generate groundtruth
  for (auto range : query_range_list) {
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = 0;
      int r_bound = range - 1;

      int search_key_range = r_bound - l_bound + 1;
      if (this->real_keys)
        search_key_range =
            this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }
  cout << "  Done!" << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

// For evaluating in Benchmark, use only 7 points in the benchmark: 0.1% 0.5% 1%
// 5% 10% 50% 100%

void DataWrapper::generateHalfBoundedQueriesAndGroundtruthBenchmark(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  vector<int> query_range_list;
  query_range_list.emplace_back(this->data_size * 0.001);
  query_range_list.emplace_back(this->data_size * 0.005);
  query_range_list.emplace_back(this->data_size * 0.01);
  query_range_list.emplace_back(this->data_size * 0.05);
  query_range_list.emplace_back(this->data_size * 0.1);
  query_range_list.emplace_back(this->data_size * 0.5);
  query_range_list.emplace_back(this->data_size);

  cout << "Generating Half Bounded Groundtruth...";
  cout << endl << "Ranges: " << endl;
  print_set(query_range_list);
  // generate groundtruth
  for (auto range : query_range_list) {
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = 0;
      int r_bound = range - 1;

      int search_key_range = r_bound - l_bound + 1;
      if (this->real_keys)
        search_key_range =
            this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }
  cout << "  Done!" << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruthBenchmark(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  vector<int> query_range_list;
  query_range_list.emplace_back(this->data_size * 0.001);
  query_range_list.emplace_back(this->data_size * 0.005);
  query_range_list.emplace_back(this->data_size * 0.01);
  query_range_list.emplace_back(this->data_size * 0.05);
  query_range_list.emplace_back(this->data_size * 0.1);
  query_range_list.emplace_back(this->data_size * 0.5);
  query_range_list.emplace_back(this->data_size);

  cout << "Generating Range Filtering Groundtruth...";
  cout << endl << "Ranges: " << endl;
  print_set(query_range_list);
  // generate groundtruth

  std::default_random_engine e;

  for (auto range : query_range_list) {
    std::uniform_int_distribution<int> u_lbound(0,
                                                this->data_size - range - 80);
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = u_lbound(e);
      int r_bound = l_bound + range - 1;
      if (range == this->data_size) {
        l_bound = 0;
        r_bound = this->data_size - 1;
      }
      int search_key_range = r_bound - l_bound + 1;
      // if (this->real_keys)
      //   search_key_range =
      //       this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }

  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}
