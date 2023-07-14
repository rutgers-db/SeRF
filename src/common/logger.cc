#include "logger.h"

// compact hnsw
void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int initial_graph_size, const int index_graph_size,
                  const string &method, const int search_ef,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size,
                  const int num_comparison, const double path_time) {
  std::ofstream file;
  file.open(path, std::ios_base::app);
  if (file) {
    file << idx << "," << l_bound << "," << r_bound << "," << range << ","
         << K_neighbor << "," << initial_graph_size << "," << index_graph_size
         << "," << method << "," << search_ef << "," << precision << ","
         << appr_ratio << "," << search_time << "," << data_size << ","
         << num_comparison << "," << path_time;
    file << "\n";
  }
  file.close();
}

// knn-first
void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int initial_graph_size, const int index_graph_size,
                  const string &method, const int search_ef,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size,
                  const size_t num_search_comparison,
                  const double out_bound_candidates,
                  const double in_bound_candidates) {
  std::ofstream file;
  file.open(path, std::ios_base::app);
  if (file) {
    file << idx << "," << l_bound << "," << r_bound << "," << range << ","
         << K_neighbor << "," << initial_graph_size << "," << index_graph_size
         << "," << method << "," << search_ef << "," << precision << ","
         << appr_ratio << "," << search_time << "," << data_size << ","
         << num_search_comparison << "," << out_bound_candidates << ","
         << in_bound_candidates;
    file << "\n";
  }
  file.close();
}

void SaveToIndexCSVRow(const string &path, const string &version,
                       const string &method, const int data_size,
                       const int initial_graph_size, const int index_graph_size,
                       const double nn_build_time, const double sort_time,
                       const double build_time, const double memory,
                       const int node_amount, const int window_count,
                       const double index_size) {
  std::ofstream file;
  file.open(path, std::ios_base::app);
  if (file) {
    file << version << "," << method << "," << data_size << ","
         << initial_graph_size << "," << index_graph_size << ","
         << nn_build_time << "," << sort_time << "," << build_time << ","
         << memory << "," << node_amount << "," << window_count << ","
         << index_size;
    file << "\n";
  }
  file.close();
}

// For range filtering, HNSW detail
void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int initial_graph_size, const int index_graph_size,
                  const string &method, const int search_ef,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size,
                  vector<int> &res, vector<float> &dists) {
  std::ofstream file;
  file.open(path, std::ios_base::app);
  if (file) {
    file << idx << "," << l_bound << "," << r_bound << "," << range << ","
         << K_neighbor << "," << initial_graph_size << "," << index_graph_size
         << "," << method << "," << search_ef << "," << precision << ","
         << appr_ratio << "," << search_time << "," << data_size << ",";

    for (auto ele : res) {
      file << ele << " ";
    }
    file << ",";
    for (auto ele : dists) {
      file << ele << " ";
    }
    file << "\n";
  }
  file.close();
}

// For PQ
void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int M_pq, const int Ks_pq, const string &method,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size) {
  std::ofstream file;
  file.open(path, std::ios_base::app);
  if (file) {
    file << idx << "," << l_bound << "," << r_bound << "," << range << ","
         << K_neighbor << "," << M_pq << "," << Ks_pq << "," << method << ","
         << precision << "," << appr_ratio << "," << search_time << ","
         << data_size;
    file << "\n";
  }
  file.close();
}