#include "utils.h"

// l2 norm
float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs,
                        const int &startDim, int lensDim) {
  float ans = 0.0;
  if (lensDim == 0) {
    lensDim = lhs.size();
  }

  for (int i = startDim; i < startDim + lensDim; ++i) {
    ans += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
  }
  return ans;
}

// l2sqr
float EuclideanDistanceSquare(const vector<float> &lhs,
                              const vector<float> &rhs) {
  float ans = 0.0;

  for (int i = 0; i < lhs.size(); ++i) {
    ans += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
  }
  return ans;
}

void testUTIL2() { cout << "hello" << endl; }

float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs) {
  return EuclideanDistance(lhs, rhs, 0, 0);
}

// t1:begin, t2:end
void AccumulateTime(timeval &t1, timeval &t2, double &val_time) {
  val_time += (t2.tv_sec - t1.tv_sec +
               (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
}

void CountTime(timeval &t1, timeval &t2, double &val_time) {
  val_time = 0;
  val_time += (t2.tv_sec - t1.tv_sec +
               (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
}

double CountTime(timeval &t1, timeval &t2) {
  double val_time = 0.0;
  val_time += (t2.tv_sec - t1.tv_sec +
               (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
  return val_time;
}

void logTime(timeval &begin, timeval &end, const string &log) {
  gettimeofday(&end, NULL);
  fprintf(stdout, ("# " + log + ": %.7fs\n").c_str(),
          end.tv_sec - begin.tv_sec +
              (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC);
};

double countPrecision(const vector<int> &truth, const vector<int> &pred) {
  double num_right = 0;
  for (auto one : truth) {
    if (find(pred.begin(), pred.end(), one) != pred.end()) {
      num_right += 1;
    }
  }
  return num_right / truth.size();
}

double countApproximationRatio(const vector<vector<float>> &raw_data,
                               const vector<int> &truth,
                               const vector<int> &pred,
                               const vector<float> &query) {
  if (pred.size() == 0) {
    return 0;
  }
  vector<float> truth_dist;
  vector<float> pred_dist;
  for (auto vec : truth) {
    truth_dist.emplace_back(EuclideanDistance(query, raw_data[vec]));
  }
  for (auto vec : pred) {
    if (vec == -1) continue;
    pred_dist.emplace_back(EuclideanDistance(query, raw_data[vec]));
  }
  if (pred_dist.size() == 0) {
    return 0;
  }
  auto max_truth = *max_element(truth_dist.begin(), truth_dist.end());
  auto max_pred = *max_element(pred_dist.begin(), pred_dist.end());
  if (pred.size() < truth.size()) {
    nth_element(truth_dist.begin(), truth_dist.begin() + pred.size() - 1,
                truth_dist.end());
    max_truth = truth_dist[pred.size() - 1];
  }
  if (max_truth != 0) return max_pred / max_truth;
  // cout << "ERROR: empty pred!" << endl;
  return -1;
}

void print_memory() {
#ifdef __linux__
  struct sysinfo memInfo;

  sysinfo(&memInfo);
  // long long totalVirtualMem = memInfo.totalram;
  // // Add other values in next statement to avoid int overflow on right hand
  // // side...
  // totalVirtualMem += memInfo.totalswap;
  // totalVirtualMem *= memInfo.mem_unit;

  // long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
  // // Add other values in next statement to avoid int overflow on right hand
  // // side...
  // virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
  // virtualMemUsed *= memInfo.mem_unit;
  // cout << "Total Virtual Memory: " << totalVirtualMem << endl;
  // cout << "Used Virtual Memory: " << virtualMemUsed << endl;

  long long totalPhysMem = memInfo.totalram;
  // Multiply in next statement to avoid int overflow on right hand side...
  totalPhysMem *= memInfo.mem_unit;

  long long physMemUsed = memInfo.totalram - memInfo.freeram;
  // Multiply in next statement to avoid int overflow on right hand side...
  physMemUsed *= memInfo.mem_unit;

  // cout << "Total Physical Memory: " << totalPhysMem << endl;
  cout << "Used Physical Memory: " << physMemUsed << endl;
#elif __APPLE__
  vm_size_t page_size;
  mach_port_t mach_port;
  mach_msg_type_number_t count;
  vm_statistics64_data_t vm_stats;

  mach_port = mach_host_self();
  count = sizeof(vm_stats) / sizeof(natural_t);
  if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
      KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
                                        (host_info64_t)&vm_stats, &count)) {
    long long free_memory = (int64_t)vm_stats.free_count * (int64_t)page_size;

    long long used_memory =
        ((int64_t)vm_stats.active_count + (int64_t)vm_stats.inactive_count +
         (int64_t)vm_stats.wire_count) *
        (int64_t)page_size;
    printf("free memory: %lld\nused memory: %lld\n", free_memory, used_memory);
  }
#endif
}

void record_memory(long long &memory) {
#ifdef __linux__
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  long long physMemUsed = memInfo.totalram - memInfo.freeram;
  physMemUsed *= memInfo.mem_unit;
  memory = physMemUsed;
#elif __APPLE__
  vm_size_t page_size;
  mach_port_t mach_port;
  mach_msg_type_number_t count;
  vm_statistics64_data_t vm_stats;

  mach_port = mach_host_self();
  count = sizeof(vm_stats) / sizeof(natural_t);
  if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
      KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
                                        (host_info64_t)&vm_stats, &count)) {
    memory = ((int64_t)vm_stats.active_count +
              (int64_t)vm_stats.inactive_count + (int64_t)vm_stats.wire_count) *
             (int64_t)page_size;
  }
#endif
}

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int k_smallest) {
  std::priority_queue<std::pair<float, int>> top_candidates;
  float lower_bound = _INT_MAX;
  for (size_t i = 0; i < dpts.size(); i++) {
    float dist = EuclideanDistanceSquare(query, dpts[i]);
    if (top_candidates.size() < k_smallest || dist < lower_bound) {
      top_candidates.push(std::make_pair(dist, i));
      if (top_candidates.size() > k_smallest) {
        top_candidates.pop();
      }

      lower_bound = top_candidates.top().first;
    }
  }
  vector<int> res;
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::reverse(res.begin(), res.end());
  return res;
}

// void evaluateKNNG(const vector<vector<int>> &gt,
//                   const vector<vector<int>> &knng, const int K, double
//                   &recall, double &precision) {
//   assert(gt.size() == knng.size());

//   double all_right = 0;
//   int knng_amount = 0;

// #pragma omp parallel for reduction(+ : all_right) reduction(+ : knng_amount)
//   for (unsigned n = 0; n < gt.size(); n++) {
//     double num_right = 0;
//     // skip first, itself
//     for (unsigned i = 1; i < K + 1; i++) {
//       int one = gt[n][i];
//       if (find(knng[n].begin(), knng[n].end(), one) != knng[n].end()) {
//         num_right += 1;
//       }
//     }
//     all_right += num_right;
//     knng_amount += knng[n].size();
//   }
//   recall = (double)all_right / (K * gt.size());
//   precision = (double)all_right / (float)knng_amount;
// }
void greedyNearest(const int query_pos, const vector<vector<float>> &dpts,
                   const int k_smallest, const int l_bound, const int r_bound) {
  vector<float> dist_arr;
  for (size_t i = l_bound; i <= r_bound; i++) {
    dist_arr.emplace_back(EuclideanDistance(dpts[query_pos], dpts[i], 0, 0));
  }
  vector<int> sorted_idxes = sort_indexes(dist_arr);

  // skip the point itself
  if (sorted_idxes[0] == query_pos) {
    sorted_idxes.erase(sorted_idxes.begin());
  }
  sorted_idxes.resize(k_smallest);
  // print_set(sorted_idxes);
}

void rangeGreedy(const vector<vector<float>> &nodes, const int k_smallest,
                 const int l_bound, const int r_bound) {
  for (size_t i = l_bound; i <= r_bound; i++) {
    greedyNearest(i, nodes, k_smallest, l_bound, r_bound);
  }
}

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int l_bound,
                          const int r_bound, const int k_smallest) {
  std::priority_queue<std::pair<float, int>> top_candidates;
  float lower_bound = _INT_MAX;
  for (size_t i = l_bound; i <= r_bound; i++) {
    float dist = EuclideanDistance(query, dpts[i]);
    if (top_candidates.size() < k_smallest || dist < lower_bound) {
      top_candidates.push(std::make_pair(dist, i));
      if (top_candidates.size() > k_smallest) {
        top_candidates.pop();
      }

      lower_bound = top_candidates.top().first;
    }
  }
  vector<int> res;
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  return res;
}

// Basic HNSW heuristic Pruning function
void heuristicPrune(const vector<vector<float>> &nodes,
                    vector<pair<int, float>> &top_candidates, const size_t M) {
  if (top_candidates.size() < M) {
    return;
  }

  std::priority_queue<std::pair<float, int>> queue_closest;
  std::vector<std::pair<float, int>> return_list;
  while (top_candidates.size() > 0) {
    queue_closest.emplace(-top_candidates.front().second,
                          top_candidates.front().first);
    top_candidates.erase(top_candidates.begin());
  }

  while (queue_closest.size()) {
    if (return_list.size() >= M) break;
    std::pair<float, int> curent_pair = queue_closest.top();
    float dist_to_query = -curent_pair.first;
    queue_closest.pop();
    bool good = true;

    for (std::pair<float, int> second_pair : return_list) {
      float curdist = EuclideanDistance(nodes.at(second_pair.second),
                                        nodes.at(curent_pair.second));
      if (curdist < dist_to_query) {
        good = false;
        break;
      }
    }
    if (good) {
      return_list.push_back(curent_pair);
    }
  }

  for (std::pair<float, int> curent_pair : return_list) {
    top_candidates.emplace_back(
        make_pair(curent_pair.second, -curent_pair.first));
  }
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