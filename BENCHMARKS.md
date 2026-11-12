# Cognivore benchmarks

## Quantized vector scans

SQ8 and PQ are standalone batch-built Python indexes. This benchmark
compares them with an exact NumPy float32 cosine scan on the same
normalized vectors and queries.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/bench_quantization.py \
  --sizes 1000 10000 50000 200000 1000000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/bench_quantization.py \
  --sizes 50000 --distribution clustered
```

The run used an Intel Core i5-1235U, Python 3.12.3, NumPy 2.5.3,
128 dimensions, seed 42 and 100 queries per size. The first distribution
uses independent Gaussian vectors normalized to the unit sphere. The
clustered distribution samples around 64 centers and normalizes the
results. PQ trains 16 subspaces with 256 centroids each for up to 12
iterations on a separate sample of 4,096 vectors from the same
distribution.

Recall@10 is the fraction of exact top-ten neighbors recovered. Query
timings include scoring and top-k selection after one warm-up query.
The float32 baseline receives pre-normalized vectors and queries. Its
build time measures copying the retained matrix and IDs. Quantized
build timings include encoding and, for PQ, codebook training. Retained
bytes include codes or vectors, IDs, cosine factors and codebooks. They
exclude temporary arrays, Python overhead, and process RSS. Reranking
rescored 40 PQ candidates using the original float32 vectors in RAM.

### Normalized Gaussian data

| Vectors | Method | Retained bytes | Build (s) | Recall@10 | p50 (ms) | p95 (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | Float32 | 520,000 | 0.000 | 1.000 | 0.023 | 0.027 |
| 1,000 | SQ8 | 140,000 | 0.002 | 0.992 | 0.049 | 0.080 |
| 1,000 | PQ | 159,072 | 0.713 | 0.412 | 0.084 | 0.105 |
| 1,000 | PQ + exact rerank | 679,072 | 0.713 | 0.730 | 0.126 | 0.167 |
| 10,000 | Float32 | 5,200,000 | 0.001 | 1.000 | 0.129 | 0.451 |
| 10,000 | SQ8 | 1,400,000 | 0.013 | 0.986 | 0.307 | 0.819 |
| 10,000 | PQ | 411,072 | 0.507 | 0.254 | 0.424 | 0.578 |
| 10,000 | PQ + exact rerank | 5,611,072 | 0.507 | 0.513 | 0.458 | 0.583 |
| 50,000 | Float32 | 26,000,000 | 0.002 | 1.000 | 1.490 | 2.997 |
| 50,000 | SQ8 | 7,000,000 | 0.040 | 0.985 | 1.528 | 2.868 |
| 50,000 | PQ | 1,531,072 | 0.697 | 0.202 | 2.014 | 2.619 |
| 50,000 | PQ + exact rerank | 27,531,072 | 0.697 | 0.417 | 2.038 | 2.549 |
| 200,000 | Float32 | 104,000,000 | 0.022 | 1.000 | 8.623 | 12.157 |
| 200,000 | SQ8 | 28,000,000 | 0.186 | 0.980 | 7.102 | 9.647 |
| 200,000 | PQ | 5,731,072 | 1.846 | 0.162 | 9.643 | 12.010 |
| 200,000 | PQ + exact rerank | 109,731,072 | 1.846 | 0.344 | 9.842 | 11.580 |
| 1,000,000 | Float32 | 520,000,000 | 0.117 | 1.000 | 42.416 | 49.966 |
| 1,000,000 | SQ8 | 140,000,000 | 0.883 | 0.985 | 40.994 | 52.066 |
| 1,000,000 | PQ | 28,131,072 | 7.936 | 0.141 | 60.680 | 82.249 |
| 1,000,000 | PQ + exact rerank | 548,131,072 | 7.936 | 0.309 | 62.928 | 82.722 |

### Clustered data

| Vectors | Method | Retained bytes | Build (s) | Recall@10 | p50 (ms) | p95 (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 50,000 | Float32 | 26,000,000 | 0.002 | 1.000 | 1.414 | 2.238 |
| 50,000 | SQ8 | 7,000,000 | 0.035 | 0.960 | 1.533 | 2.065 |
| 50,000 | PQ | 1,531,072 | 0.784 | 0.060 | 2.335 | 2.652 |
| 50,000 | PQ + exact rerank | 27,531,072 | 0.784 | 0.196 | 2.385 | 2.654 |

At one million vectors, SQ8 retained about 27% of the float32 index
arrays and recovered 98.5% of the exact top ten. PQ retained about 5.4%
with this configuration and missed most exact neighbors. Rescoring can
recover only neighbors present in the candidate set and retaining the
float32 vectors in RAM removes the memory savings. The clustered case
also showed substantial PQ recall loss. Evaluate real embedding
collections and tune the compression before using PQ for retrieval.

These tables measure standalone scans on synthetic data. Cross-backend
comparisons with Faiss, hnswlib, Chroma and Qdrant require a separate
benchmark.
