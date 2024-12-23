# SeRF

This repo is the implementation of [SeRF: Segment Graph for Range Filtering Approximate Nearest Neighbor Search](https://dl.acm.org/doi/10.1145/3639324).

| file | description |
|:--:|:--:|
| segment_graph_1d.h | SegmentGraph for halfbounded query |
| segment_graph_2d.h | 2D-SegmentGraph for arbitrary query |

> [!NOTE]
> [09/09/2024] We've optimized compile options and added SIMD support. By default, construction and search operations now run in parallel with SIMD, resulting in an approximate 1.5x boost in query speed.

## Quick Start

### Compile and Run

SeRF can be built from source using CMake. The only dependencies required are C++11 and OpenMP.

We have tested the following build instructions on macOS (ARM) and Linux (x86-64):

```bash
mkdir build && cd build
cmake ..
make
```

Running example benchmark on DEEP dataset:

```bash
./benchmark/serf_halfbound -N 10000 -dataset_path [path_to_deep_base.fvecs] -query_path [path_to_deep_query.fvecs]
./benchmark/serf_arbitrary -N 10000 -dataset_path [path_to_deep_base.fvecs] -query_path [path_to_deep_query.fvecs]

# Running on sample dataset under "sample_data"
./benchmark/serf_halfbound -N 10000 -dataset_path ../sample_data/deep_10k.fvecs -query_path ../sample_data/deep_query.fvecs
./benchmark/serf_arbitrary -N 10000 -dataset_path ../sample_data/deep_10k.fvecs -query_path ../sample_data/deep_query.fvecs
```

Parameters:

- `-dataset_path`: The base dataset path for indexing, pre-sorted by search key
- `-query_path`: The query vectors path
- `-N`: The top-N number of vector using for indexing, load all vectors if not specify

Optional Parameters:

- `-dataset`: The dataset name, supported values: `deep,yt8m-audio,wiki-image`
- `-index_k`: The maximum number of neighbors of HNSW index.
- `-ef_con`: ef_construction for HNSW index.
- `-ef_max`: The maximum number of neighbors for building SeRF.

`index_k1`, `ef_con`, `ef_max`, `ef_search` support multiple values, separating by `,`.

Sample command:

```bash
./benchmark/serf_arbitrary -N 10000 -dataset_path ../sample_data/deep_10k.fvecs -query_path ../sample_data/deep_query.fvecs -ef_search 40,60,80,100 -index_k 16
```

We hardcoded some parameters for the ease of demonstration, you can change them in the code and recompile.

## Dataset

| Dataset | Data type | Dimensions | Search Key |
| :- | :-: | :-: | :-: |
| [DEEP](http://sites.skoltech.ru/compvision/noimi/) | float | 96 | Synthetic |
| [Youtube-Audio](https://research.google.com/youtube8m/download.html) | float | 128 | Video Release Time |
| [WIT-Image](https://www.kaggle.com/c/wikipedia-image-caption/overview) | float | 1024 | Image Size |

<!-- - [DEEP](http://sites.skoltech.ru/compvision/noimi/): Each point is assigned a random number as the synthetic key.

- [Youtube-Audio](https://research.google.com/youtube8m/download.html): Video release time as the search key.

- [WIT-Image](https://www.kaggle.com/c/wikipedia-image-caption/overview): Size of the image as the search key. -->

## Reference

```text
@article{SeRF,
author = {Zuo, Chaoji and Qiao, Miao and Zhou, Wenchao and Li, Feifei and Deng, Dong},
title = {SeRF: Segment Graph for Range-Filtering Approximate Nearest Neighbor Search},
year = {2024},
issue_date = {February 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {2},
number = {1},
url = {https://doi.org/10.1145/3639324},
doi = {10.1145/3639324}
}
```
