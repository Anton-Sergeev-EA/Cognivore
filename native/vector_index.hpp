// Cognivore native vector index
//
// A small, dependency-free vector search core written in C++ and exposed to
// Python via pybind11. Two indexes are provided:
//
//   * FlatIndex - exact brute-force search over cosine similarity, with a
//     hand-rolled dot-product kernel that auto-vectorizes well (and uses
//     AVX2 intrinsics when the compiler target supports them).
//   * NSWIndex  - an approximate nearest-neighbour index based on a
//     single-layer Navigable Small World graph (the same core idea behind
//     the bottom layer of HNSW). Insertion greedily connects each new point
//     to the M best candidates found by a graph walk from a random entry
//     point; search performs the same greedy walk with a candidate beam of
//     width `ef`.
//
// Both indexes store vectors contiguously in a single buffer to keep cache
// behaviour predictable and to make save/load trivial (a flat binary dump).
//
// This is intentionally a *teaching-grade* implementation: it favours
// readability and correctness over squeezing out every last percent of
// throughput, but the design (contiguous storage, no per-vector heap
// allocation on the hot path, an explicit SIMD kernel) is the same shape
// used by production ANN libraries such as Faiss and hnswlib.
#pragma once

#include <cstdint>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cognivore {

using VectorId = int64_t;

struct SearchResult {
    VectorId id;
    float score; // cosine similarity, higher is better
};

/// Computes the dot product of two equal-length, L2-normalized float
/// vectors. Because both inputs are pre-normalized, the dot product *is*
/// the cosine similarity - no division needed on the hot path.
float dot_product(const float* a, const float* b, size_t dim) noexcept;

/// In-place L2 normalization of a vector. No-op on an all-zero vector.
void normalize(float* v, size_t dim) noexcept;

/// Exact brute-force cosine-similarity index.
class FlatIndex {
public:
    explicit FlatIndex(size_t dim);

    /// Adds one vector (copied and normalized internally) under `id`.
    void add(VectorId id, const std::vector<float>& vector);

    /// Returns the top-k most similar vectors to the (normalized) query.
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) const;

    size_t size() const noexcept { return ids_.size(); }
    size_t dim() const noexcept { return dim_; }

    /// Serializes to a flat binary blob (dim, count, ids[], vectors[]).
    std::vector<uint8_t> serialize() const;
    static FlatIndex deserialize(const std::vector<uint8_t>& blob);

private:
    size_t dim_;
    std::vector<VectorId> ids_;
    std::vector<float> data_; // row-major, size() == ids_.size() * dim_
};

/// Approximate nearest-neighbour index using a single-layer Navigable Small
/// World graph over cosine similarity.
class NSWIndex {
public:
    /// `m` is the number of bidirectional edges created per inserted node;
    /// `ef_construction` is the beam width used while searching for
    /// neighbours at insertion time (higher = better graph quality, slower
    /// build).
    explicit NSWIndex(size_t dim, size_t m = 16, size_t ef_construction = 200);

    void add(VectorId id, const std::vector<float>& vector);

    /// Approximate top-k search. `ef` is the beam width used for this
    /// query; larger values trade speed for recall. `ef` is clamped to be
    /// at least `k`.
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k, size_t ef) const;

    size_t size() const noexcept { return ids_.size(); }
    size_t dim() const noexcept { return dim_; }

    std::vector<uint8_t> serialize() const;
    static NSWIndex deserialize(const std::vector<uint8_t>& blob);

private:
    // Greedy beam search from a few random entry points over the graph as
    // it currently exists. Returns up to `ef` (internal_idx, score) pairs
    // sorted by descending similarity. `query` must already be normalized.
    std::vector<std::pair<uint32_t, float>> beam_search(const std::vector<float>& query,
                                                          size_t ef) const;

    // Keeps a node's neighbour list bounded to roughly 2*m_ edges by
    // discarding the least-similar ones. Called after a node gains a new
    // edge as a side effect of a later insertion.
    void prune_edges(uint32_t node_idx);

    const float* vector_at(size_t internal_idx) const noexcept {
        return data_.data() + internal_idx * dim_;
    }

    size_t dim_;
    size_t m_;
    size_t ef_construction_;
    std::vector<VectorId> ids_;                 // internal index -> external id
    std::vector<float> data_;                   // internal index -> normalized vector
    std::vector<std::vector<uint32_t>> edges_;   // internal index -> neighbour internal indices
    mutable std::mt19937 rng_{42};
};

} // namespace cognivore
