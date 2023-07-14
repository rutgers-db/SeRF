/**
 * @file reader.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Read Vector data
 * @date 2023-04-21
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::string;
using std::vector;

// Interface (abstract basic class) of iterative reader
class I_ItrReader {
 public:
  virtual ~I_ItrReader() {}
  virtual bool IsEnd() = 0;
  virtual std::vector<float> Next() = 0;
};

// Iterative reader for fvec file
class FvecsItrReader : I_ItrReader {
 public:
  FvecsItrReader(std::string filename);
  bool IsEnd();
  std::vector<float> Next();

 private:
  FvecsItrReader();  // prohibit default construct
  std::ifstream ifs;
  std::vector<float> vec;  // store the next vec
  bool eof_flag;
};

// Iterative reader for bvec file
class BvecsItrReader : I_ItrReader {
 public:
  BvecsItrReader(std::string filename);
  bool IsEnd();
  std::vector<float> Next();  // Read bvec, but return vec<float>
 private:
  BvecsItrReader();  // prohibit default construct
  std::ifstream ifs;
  std::vector<float> vec;  // store the next vec
  bool eof_flag;
};

// Proxy class
class ItrReader {
 public:
  // ext must be "fvecs" or "bvecs"
  ItrReader(std::string filename, std::string ext);
  ~ItrReader();

  bool IsEnd();
  std::vector<float> Next();

 private:
  ItrReader();
  I_ItrReader *m_reader;
};

// Wrapper. Read top-N vectors
// If top_n = -1, then read all vectors
std::vector<std::vector<float>> ReadTopN(std::string filename, std::string ext,
                                         int top_n = -1);

void ReadMatFromTxt(const std::string &path,
                    std::vector<std::vector<float>> &data,
                    const int length_limit);

void ReadMatFromTxtTwitter(const std::string &path,
                           std::vector<std::vector<float>> &data,
                           const int length_limit);
void ReadMatFromTsv(const std::string &path,
                    std::vector<std::vector<float>> &data,
                    const int length_limit);

void ReadDataWrapper(vector<vector<float>> &raw_data, vector<int> &search_keys,
                     const string &dataset, string &dataset_path,
                     const int item_num);
                     
void ReadDataWrapper(const string &dataset, string &dataset_path,
                     vector<vector<float>> &raw_data, const int data_size,
                     string &query_path, vector<vector<float>> &querys,
                     const int query_size, vector<int> &search_keys);

void ReadDataWrapper(const string &dataset, string &dataset_path,
                     vector<vector<float>> &raw_data, const int data_size);

int YT8M2Int(const string id);
void ReadMatFromTsvYT8M(const string &path, vector<vector<float>> &data,
                        vector<int> &search_keys, const int length_limit);

void ReadGroundtruthQuery(std::vector<std::vector<int>> &gt,
                          std::vector<std::pair<int, int>> &query_ranges,
                          std::vector<int> &query_ids, std::string gt_path);