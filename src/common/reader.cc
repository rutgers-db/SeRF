#include "reader.h"

#include <assert.h>

using std::vector;
FvecsItrReader::FvecsItrReader(std::string filename) {
  ifs.open(filename, std::ios::binary);
  assert(ifs.is_open());
  Next();
}

bool FvecsItrReader::IsEnd() { return eof_flag; }

std::vector<float> FvecsItrReader::Next() {
  std::vector<float> prev_vec = vec;  // return the currently stored vec
  int D;
  if (ifs.read((char *)&D, sizeof(int))) {  // read "D"
    // Then, read a D-dim vec
    vec.resize(D);                                    // allocate D-dim
    ifs.read((char *)vec.data(), sizeof(float) * D);  // Read D * float.
    eof_flag = false;
  } else {
    vec.clear();
    eof_flag = true;
  }
  return prev_vec;
}

BvecsItrReader::BvecsItrReader(std::string filename) {
  ifs.open(filename, std::ios::binary);
  assert(ifs.is_open());
  Next();
}

bool BvecsItrReader::IsEnd() { return eof_flag; }

std::vector<float> BvecsItrReader::Next() {
  std::vector<float> prev_vec = vec;  // return the currently stored vec
  int D;
  if (ifs.read((char *)&D, sizeof(int))) {  // read "D"
    // Then, read a D-dim vec
    vec.resize(D);  // allocate D-dim
    std::vector<unsigned char> buff(D);

    assert(ifs.read((char *)buff.data(),
                    sizeof(unsigned char) * D));  // Read D * uchar.

    // Convert uchar to float
    for (int d = 0; d < D; ++d) {
      vec[d] = static_cast<float>(buff[d]);
    }

    eof_flag = false;
  } else {
    vec.clear();
    eof_flag = true;
  }
  return prev_vec;
}

ItrReader::ItrReader(std::string filename, std::string ext) {
  if (ext == "fvecs") {
    m_reader = (I_ItrReader *)new FvecsItrReader(filename);
  } else if (ext == "bvecs") {
    m_reader = (I_ItrReader *)new BvecsItrReader(filename);
  } else {
    std::cerr << "Error: strange ext type: " << ext << "in ItrReader"
              << std::endl;
    exit(1);
  }
}

ItrReader::~ItrReader() { delete m_reader; }

bool ItrReader::IsEnd() { return m_reader->IsEnd(); }

std::vector<float> ItrReader::Next() { return m_reader->Next(); }

std::vector<std::vector<float>> ReadTopN(std::string filename, std::string ext,
                                         int top_n) {
  std::vector<std::vector<float>> vecs;
  if (top_n != -1) {
    vecs.reserve(top_n);
  }
  ItrReader reader(filename, ext);
  while (!reader.IsEnd()) {
    if (top_n != -1 && top_n <= (int)vecs.size()) {
      break;
    }
    vecs.emplace_back(reader.Next());
  }
  return vecs;
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void ReadFvecsTopN(const std::string &file_path,
                   std::vector<std::vector<float>> &data, const uint32_t N,
                   const int num_dimensions) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());

  data.resize(N);
  std::vector<double> buff(num_dimensions);
  int counter = 0;
  while ((counter < N) &&
         (ifs.read((char *)buff.data(), num_dimensions * sizeof(double)))) {
    std::vector<float> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<float>(buff[d]);
    }
    data[counter++] = std::move(row);
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file. Skip some nodes, for reading querys
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void ReadFvecsSkipTop(const std::string &file_path,
                      std::vector<std::vector<float>> &data, const uint32_t N,
                      const int num_dimensions, const int skip_num) {
  std::cout << "Query Start From Position: " << skip_num << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  ifs.seekg(num_dimensions * sizeof(double) * skip_num);

  data.resize(N);
  std::vector<double> buff(num_dimensions);
  int counter = 0;
  while ((counter < N) &&
         ifs.read((char *)buff.data(), num_dimensions * sizeof(double))) {
    std::vector<float> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<float>(buff[d]);
    }
    data[counter++] = std::move(row);
  }

  ifs.close();
}

/// @brief Reading metadata information, stored in uint32_t format
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void ReadIvecsTopN(const std::string &file_path, std::vector<int> &keys,
                   const uint32_t N, const int num_dimensions,
                   const int position) {
  std::cout << "Reading Keys: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());

  keys.resize(N);
  std::vector<uint64_t> buff(num_dimensions);
  int counter = 0;
  while ((counter < N) &&
         ifs.read((char *)buff.data(), num_dimensions * sizeof(uint64_t))) {
    keys[counter++] = static_cast<uint64_t>(buff[position]);
  }

  ifs.close();
  std::cout << "Finish Reading Keys" << endl;
}

void ReadDataWrapper(vector<vector<float>> &raw_data, vector<int> &search_keys,
                     const string &dataset, string &dataset_path,
                     const int item_num) {
  raw_data.clear();
  if (dataset == "glove") {
    ReadMatFromTxtTwitter(dataset_path, raw_data, item_num);
  } else if (dataset == "ml25m") {
    ReadMatFromTxt(dataset_path, raw_data, item_num);
  } else if (dataset == "sift") {
    raw_data = ReadTopN(dataset_path, "bvecs", item_num);
  } else if (dataset == "biggraph") {
    ReadMatFromTsv(dataset_path, raw_data, item_num);
  } else if (dataset == "local") {
    raw_data = ReadTopN(dataset_path, "fvecs", item_num);
  } else if (dataset == "deep") {
    raw_data = ReadTopN(dataset_path, "fvecs", item_num);
  } else if (dataset == "deep10m") {
    raw_data = ReadTopN(dataset_path, "fvecs", item_num);
  } else if (dataset == "yt8m") {
    ReadMatFromTsvYT8M(dataset_path, raw_data, search_keys, item_num);
  } else {
    std::cerr << "Wrong Datset!" << endl;
    assert(false);
  }
}

// load data and querys
void ReadDataWrapper(const string &dataset, string &dataset_path,
                     vector<vector<float>> &raw_data, const int data_size,
                     string &query_path, vector<vector<float>> &querys,
                     const int query_size, vector<int> &search_keys) {
  raw_data.clear();
  if (dataset == "glove" || dataset == "glove25" || dataset == "glove50" ||
      dataset == "glove100" || dataset == "glove200") {
    ReadMatFromTxtTwitter(dataset_path, raw_data, data_size);
  } else if (dataset == "ml25m") {
    ReadMatFromTxt(dataset_path, raw_data, data_size);
  } else if (dataset == "sift") {
    raw_data = ReadTopN(dataset_path, "bvecs", data_size);
    querys = ReadTopN(query_path, "bvecs", query_size);
  } else if (dataset == "biggraph") {
    ReadMatFromTsv(dataset_path, raw_data, data_size);
  } else if (dataset == "local") {
    cout << dataset_path << endl;
    raw_data = ReadTopN(dataset_path, "fvecs", data_size);
  } else if (dataset == "deep") {
    raw_data = ReadTopN(dataset_path, "fvecs", data_size);
    querys = ReadTopN(query_path, "fvecs", query_size);
  } else if (dataset == "deep10m") {
    raw_data = ReadTopN(dataset_path, "fvecs", data_size);
  } else if (dataset == "yt8m") {
    ReadMatFromTsvYT8M(dataset_path, raw_data, search_keys, data_size);
  } else if (dataset == "yt8m-video") {
    ReadFvecsTopN(dataset_path, raw_data, data_size, 1024);
    ReadFvecsTopN(query_path, querys, query_size, 1024);
  } else if (dataset == "yt8m-audio") {
    ReadFvecsTopN(dataset_path, raw_data, data_size, 128);
    ReadFvecsTopN(query_path, querys, query_size, 128);

  } else if (dataset == "wiki-image") {
    ReadFvecsTopN(dataset_path, raw_data, data_size, 2048);
    ReadFvecsTopN(query_path, querys, query_size, 2048);

  }

  else {
    std::cerr << "Wrong Datset!" << endl;
    assert(false);
  }
}

void Split(std::string &s, std::string &delim, std::vector<std::string> *ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret->push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (index - last > 0) {
    ret->push_back(s.substr(last, index - last));
  }
}

// load txt matrix data
void ReadMatFromTxt(const string &path, vector<vector<float>> &data,
                    const int length_limit = -1) {
  ifstream infile;
  string bline;
  string delim = " ";
  int numCols = 0;
  infile.open(path, ios::in);
  if (getline(infile, bline, '\n')) {
    vector<string> ret;
    Split(bline, delim, &ret);
    numCols = ret.size();
  }
  infile.close();
  // cout << "Reading " << path << " ..." << endl;
  // cout << "# of columns: " << numCols << endl;

  int counter = 0;
  if (length_limit == -1) counter = -9999999;
  // TODO: read sparse matrix
  infile.open(path, ios::in);
  while (getline(infile, bline, '\n')) {
    if (counter >= length_limit) break;
    counter++;

    vector<string> ret;
    Split(bline, delim, &ret);
    vector<float> arow(numCols);
    assert(ret.size() == numCols);
    for (int i = 0; i < ret.size(); i++) {
      arow[i] = static_cast<float>(stod(ret[i]));
    }
    data.emplace_back(arow);
  }
  infile.close();
  // cout << "# of rows: " << data.size() << endl;
}

void ReadMatFromTxtTwitter(const string &path, vector<vector<float>> &data,
                           const int length_limit = -1) {
  ifstream infile;
  string bline;
  string delim = " ";
  int numCols = 0;
  infile.open(path, ios::in);
  if (getline(infile, bline, '\n')) {
    vector<string> ret;
    Split(bline, delim, &ret);
    numCols = ret.size() - 1;
  }
  infile.close();
  cout << "Reading " << path << " ..." << endl;

  cout << "# of columns: " << numCols << endl;

  int counter = 0;
  if (length_limit == -1) counter = -9999999;
  // TODO: read sparse matrix
  infile.open(path, ios::in);
  while (getline(infile, bline, '\n')) {
    if (counter >= length_limit) break;
    counter++;

    vector<string> ret;
    Split(bline, delim, &ret);
    vector<float> arow(numCols);
    assert(ret.size() == numCols + 1);
    for (int i = 1; i < ret.size(); i++) {
      arow[i - 1] = static_cast<float>(stod(ret[i]));
    }
    data.emplace_back(arow);
  }
  infile.close();
  cout << "# of rows: " << data.size() << endl;
}

void ReadMatFromTsv(const string &path, vector<vector<float>> &data,
                    const int length_limit = -1) {
  ifstream infile;
  string bline;
  string delim = "\t";
  int numCols = 0;
  infile.open(path, ios::in);
  getline(infile, bline, '\n');
  if (getline(infile, bline, '\n')) {
    vector<string> ret;
    Split(bline, delim, &ret);
    numCols = ret.size();
  }
  infile.close();
  cout << "Reading " << path << " ..." << endl;
  cout << "# of columns: " << numCols << endl;

  int counter = 0;
  if (length_limit == -1) counter = -9999999;
  infile.open(path, ios::in);
  // skip the first line
  getline(infile, bline, '\n');
  while (getline(infile, bline, '\n')) {
    if (counter >= length_limit) break;
    counter++;

    vector<string> ret;
    Split(bline, delim, &ret);
    vector<float> arow(numCols - 1);
    assert(ret.size() == numCols);
    for (int i = 0; i < ret.size() - 1; i++) {
      arow[i] = static_cast<float>(stod(ret[i + 1]));
    }
    data.emplace_back(arow);
  }
  infile.close();
  cout << "# of rows: " << data.size() << endl;
}

int YT8M2Int(const string id) {
  int res = 0;
  for (size_t i = 0; i < 4; i++) {
    res *= 100;
    res += (int)id[i] - 38;
  }
  return res;
}

void ReadMatFromTsvYT8M(const string &path, vector<vector<float>> &data,
                        vector<int> &search_keys, const int length_limit) {
  ifstream infile;
  string bline;
  string delim = ",";
  int numCols = 0;
  infile.open(path, ios::in);
  getline(infile, bline, '\n');
  if (getline(infile, bline, '\n')) {
    vector<string> ret;
    Split(bline, delim, &ret);
    numCols = ret.size();
  }
  infile.close();
  cout << "Reading " << path << " ..." << endl;
  cout << "# of columns: " << numCols << endl;

  int counter = 0;
  if (length_limit == -1) counter = -9999999;
  infile.open(path, ios::in);
  string delim_embed = " ";

  while (getline(infile, bline, '\n')) {
    if (counter >= length_limit) break;
    counter++;

    vector<string> ret;
    Split(bline, delim, &ret);
    assert(ret.size() == numCols);

    // str 'id' to int 'id'
    // int one_search_key = YT8M2Int(ret[0]);
    int one_search_key = (int)stod(ret[1]);

    // add embedding
    string embedding_str = ret[2];
    vector<string> embedding_vec;
    vector<float> arow(1024);
    Split(embedding_str, delim_embed, &embedding_vec);
    assert(embedding_vec.size() == 1024);
    for (int i = 0; i < embedding_vec.size() - 1; i++) {
      arow[i] = static_cast<float>(stod(embedding_vec[i + 1]));
    }
    search_keys.emplace_back(one_search_key);
    data.emplace_back(arow);
  }
  infile.close();
  cout << "# of rows: " << data.size() << endl;
}

void ReadMatFromTsvYT8M(const string &path, vector<vector<float>> &data,
                        const int length_limit) {
  ifstream infile;
  string bline;
  string delim = ",";
  int numCols = 0;
  infile.open(path, ios::in);
  getline(infile, bline, '\n');
  if (getline(infile, bline, '\n')) {
    vector<string> ret;
    Split(bline, delim, &ret);
    numCols = ret.size();
  }
  infile.close();
  cout << "Reading " << path << " ..." << endl;
  cout << "# of columns: " << numCols << endl;

  int counter = 0;
  if (length_limit == -1) counter = -9999999;
  infile.open(path, ios::in);
  string delim_embed = " ";

  while (getline(infile, bline, '\n')) {
    if (counter >= length_limit) break;
    counter++;

    vector<string> ret;
    Split(bline, delim, &ret);
    assert(ret.size() == numCols);

    // add embedding
    string embedding_str = ret[2];
    vector<string> embedding_vec;
    vector<float> arow(1024);
    Split(embedding_str, delim_embed, &embedding_vec);
    assert(embedding_vec.size() == 1024);
    for (int i = 0; i < embedding_vec.size() - 1; i++) {
      arow[i] = static_cast<float>(stod(embedding_vec[i + 1]));
    }
    data.emplace_back(arow);
  }
  infile.close();
  cout << "# of rows: " << data.size() << endl;
}

void ReadGroundtruthQuery(vector<vector<int>> &gt,
                          vector<std::pair<int, int>> &query_ranges,
                          vector<int> &query_ids, string gt_path) {
  ifstream infile;
  string bline;
  string delim = ",";
  string space_delim = " ";

  int numCols = 0;
  infile.open(gt_path, ios::in);
  assert(infile.is_open());

  int counter = 0;
  while (getline(infile, bline, '\n')) {
    counter++;
    vector<int> one_gt;
    std::pair<int, int> one_range;
    int one_id;
    vector<string> ret;
    Split(bline, delim, &ret);
    one_id = std::stoi(ret[0]);
    one_range.first = std::stoi(ret[1]);
    one_range.second = std::stoi(ret[2]);
    vector<string> str_gt;
    Split(ret[7], space_delim, &str_gt);
    str_gt.pop_back();
    for (auto ele : str_gt) {
      one_gt.emplace_back(std::stoi(ele));
    }
    gt.emplace_back(one_gt);
    query_ranges.emplace_back(one_range);
    query_ids.emplace_back(one_id);
  }
}

void fvecs2csv(const string &output_path, const vector<vector<float>> &nodes) {
  std::ofstream file;
  file.open(output_path, std::ios_base::app);
  for (auto row : nodes) {
    if (file) {
      for (auto ele : row) {
        file << ele << " ";
      }
      file << "\n";
    }
  }
  file.close();
}

// load data and querys
void ReadDataWrapper(const string &dataset, string &dataset_path,
                     vector<vector<float>> &raw_data, const int data_size) {
  raw_data.clear();
  if (dataset == "glove" || dataset == "glove25" || dataset == "glove50" ||
      dataset == "glove100" || dataset == "glove200") {
    ReadMatFromTxtTwitter(dataset_path, raw_data, data_size);
  } else if (dataset == "ml25m") {
    ReadMatFromTxt(dataset_path, raw_data, data_size);
  } else if (dataset == "sift") {
    raw_data = ReadTopN(dataset_path, "bvecs", data_size);
  } else if (dataset == "biggraph") {
    ReadMatFromTsv(dataset_path, raw_data, data_size);
  } else if (dataset == "local") {
    raw_data = ReadTopN(dataset_path, "fvecs", data_size);
  } else if (dataset == "deep") {
    raw_data = ReadTopN(dataset_path, "fvecs", data_size);
  } else if (dataset == "deep10m") {
    raw_data = ReadTopN(dataset_path, "fvecs", data_size);
  } else if (dataset == "yt8m") {
    ReadMatFromTsvYT8M(dataset_path, raw_data, data_size);
  } else {
    std::cerr << "Wrong Datset!" << endl;
    assert(false);
  }
}