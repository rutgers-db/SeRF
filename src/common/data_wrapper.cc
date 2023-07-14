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
  vector<size_t> index_permutation;
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
  while (init_range < this->data_size) {
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

  cout << "Generating Groundtruth...";
  // generate groundtruth

  for (auto range : query_range_list) {
    if (range > this->data_size - 100) {
      break;
    }
    std::uniform_int_distribution<int> u_lbound(0,
                                                this->data_size - range - 80);
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = u_lbound(e);
      int r_bound = l_bound + range;
      int search_key_range = r_bound - l_bound;
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
  while (init_range < this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * 0.05;
  }

  cout << "Generating Half Bounded Groundtruth...";
  // generate groundtruth
  for (auto range : query_range_list) {
    if (range > this->data_size - 100) {
      break;
    }
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = 0;
      int r_bound = range;

      int search_key_range = r_bound - l_bound;
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
  cout << "Loading Groundtruth from" << gt_path << "...";
  ReadGroundtruthQuery(this->groundtruth, this->query_ranges, this->query_ids,
                       gt_path);
  cout << "    Done!" << endl;
}