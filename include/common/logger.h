/**
 * @file logger.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief for output exp result to csv files
 * @date 2023-06-19
 *
 * @copyright Copyright (c) 2023
 */
#pragma once

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

// compact hnsw

void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int initial_graph_size, const int index_graph_size,
                  const string &method, const int search_ef,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size,
                  const int num_comparison, const double path_time);

// knn-first
void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int initial_graph_size, const int index_graph_size,
                  const string &method, const int search_ef,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size,
                  const size_t num_search_comparison,
                  const double out_bound_candidates,
                  const double in_bound_candidates);

void SaveToIndexCSVRow(const string &path, const string &version,
                       const string &method, const int data_size,
                       const int initial_graph_size, const int index_graph_size,
                       const double nn_build_time, const double sort_time,
                       const double build_time, const double memory,
                       const int node_amount, const int window_count,
                       const double index_size);

// For range filtering, HNSW detail
void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int initial_graph_size, const int index_graph_size,
                  const string &method, const int search_ef,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size,
                  vector<int> &res, vector<float> &dists);

// For PQ
void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int range, const int K_neighbor,
                  const int M_pq, const int Ks_pq, const string &method,
                  const double &precision, const double &appr_ratio,
                  const double &search_time, const int data_size);