#include "vector_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <stdexcept>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define COGNIVORE_USE_AVX2 1
#endif

namespace cognivore {

float dot_product(const float* a, const float* b, size_t dim) noexcept {
#if defined(COGNIVORE_USE_AVX2)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    size_t simd_end = dim - (dim % 8);
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, acc);
    float sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] +
                buffer[7];
    for (; i < dim; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

void normalize(float* v, size_t dim) noexcept {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum_sq += v[i] * v[i];
    if (sum_sq <= 1e-20f) return;
    float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (size_t i = 0; i < dim; ++i) v[i] *= inv_norm;
}

// ----------------------------------------------------------------------
// FlatIndex
// ----------------------------------------------------------------------

FlatIndex::FlatIndex(size_t dim) : dim_(dim) {}

void FlatIndex::add(VectorId id, const std::vector<float>& vector) {
    if (vector.size() != dim_) throw std::invalid_argument("vector dimension mismatch");
    ids_.push_back(id);
    size_t offset = data_.size();
    data_.resize(offset + dim_);
    std::memcpy(data_.data() + offset, vector.data(), dim_ * sizeof(float));
    normalize(data_.data() + offset, dim_);
}

std::vector<SearchResult> FlatIndex::search(const std::vector<float>& query, size_t k) const {
    if (query.size() != dim_) throw std::invalid_argument("query dimension mismatch");
    std::vector<float> q(query);
    normalize(q.data(), dim_);

    // Scoring every stored vector against the query is embarrassingly
    // parallel (each iteration only reads shared data and writes its own
    // slot), so we hand it to OpenMP when the extension was built with it
    // (-fopenmp on Linux/Windows; see setup.py). Without OpenMP enabled at
    // compile time this `#pragma` is simply ignored by the compiler and the
    // loop runs sequentially -- identical results either way, just slower.
    std::vector<SearchResult> all(ids_.size());
    const long n = static_cast<long>(ids_.size());
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        size_t idx = static_cast<size_t>(i);
        float score = dot_product(q.data(), data_.data() + idx * dim_, dim_);
        all[idx] = {ids_[idx], score};
    }
    k = std::min(k, all.size());
    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                       [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
    all.resize(k);
    return all;
}

std::vector<uint8_t> FlatIndex::serialize() const {
    std::vector<uint8_t> buf;
    auto push = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), b, b + n);
    };
    uint64_t dim64 = dim_, count64 = ids_.size();
    push(&dim64, sizeof(dim64));
    push(&count64, sizeof(count64));
    push(ids_.data(), ids_.size() * sizeof(VectorId));
    push(data_.data(), data_.size() * sizeof(float));
    return buf;
}

FlatIndex FlatIndex::deserialize(const std::vector<uint8_t>& blob) {
    size_t offset = 0;
    auto pull = [&](void* p, size_t n) {
        if (offset + n > blob.size()) throw std::runtime_error("corrupt FlatIndex blob");
        std::memcpy(p, blob.data() + offset, n);
        offset += n;
    };
    uint64_t dim64 = 0, count64 = 0;
    pull(&dim64, sizeof(dim64));
    pull(&count64, sizeof(count64));
    FlatIndex idx(static_cast<size_t>(dim64));
    idx.ids_.resize(static_cast<size_t>(count64));
    pull(idx.ids_.data(), idx.ids_.size() * sizeof(VectorId));
    idx.data_.resize(static_cast<size_t>(count64) * static_cast<size_t>(dim64));
    pull(idx.data_.data(), idx.data_.size() * sizeof(float));
    return idx;
}

// ----------------------------------------------------------------------
// NSWIndex
// ----------------------------------------------------------------------

namespace {
struct Item {
    float score;
    uint32_t idx;
};
// std::priority_queue is a max-heap by default when using operator<, so a
// plain "a.score < b.score" comparator gives us a max-heap (top = best).
struct ByScoreMax {
    bool operator()(const Item& a, const Item& b) const { return a.score < b.score; }
};
// Reversing the comparison gives a min-heap (top = worst), which is what we
// want for the running result set so we can cheaply evict the worst item.
struct ByScoreMin {
    bool operator()(const Item& a, const Item& b) const { return a.score > b.score; }
};
} // namespace

NSWIndex::NSWIndex(size_t dim, size_t m, size_t ef_construction)
    : dim_(dim), m_(m), ef_construction_(ef_construction) {}

std::vector<std::pair<uint32_t, float>> NSWIndex::beam_search(const std::vector<float>& query,
                                                                size_t ef) const {
    if (ids_.empty()) return {};
    ef = std::max<size_t>(ef, 1);

    std::priority_queue<Item, std::vector<Item>, ByScoreMax> candidates; // frontier, best first
    std::priority_queue<Item, std::vector<Item>, ByScoreMin> results;   // kept-so-far, worst on top
    std::unordered_set<uint32_t> visited;

    size_t num_entries = std::min<size_t>(3, ids_.size());
    std::uniform_int_distribution<size_t> dist(0, ids_.size() - 1);
    for (size_t r = 0; r < num_entries; ++r) {
        uint32_t entry = static_cast<uint32_t>(dist(rng_));
        if (!visited.insert(entry).second) continue;
        float score = dot_product(query.data(), vector_at(entry), dim_);
        candidates.push({score, entry});
        results.push({score, entry});
        if (results.size() > ef) results.pop();
    }

    while (!candidates.empty()) {
        Item current = candidates.top();
        candidates.pop();
        // Once the best remaining candidate can't beat our worst kept
        // result, no further expansion can improve the answer (everything
        // reachable from here is explored in roughly descending quality).
        if (results.size() >= ef && current.score < results.top().score) {
            break;
        }
        for (uint32_t neighbor : edges_[current.idx]) {
            if (!visited.insert(neighbor).second) continue;
            float score = dot_product(query.data(), vector_at(neighbor), dim_);
            if (results.size() < ef || score > results.top().score) {
                candidates.push({score, neighbor});
                results.push({score, neighbor});
                if (results.size() > ef) results.pop();
            }
        }
    }

    std::vector<std::pair<uint32_t, float>> out;
    out.reserve(results.size());
    while (!results.empty()) {
        out.emplace_back(results.top().idx, results.top().score);
        results.pop();
    }
    std::reverse(out.begin(), out.end()); // heap pops ascending -> flip to descending
    return out;
}

void NSWIndex::prune_edges(uint32_t node_idx) {
    auto& nbrs = edges_[node_idx];
    size_t max_degree = m_ * 2;
    if (nbrs.size() <= max_degree) return;

    std::vector<Item> scored;
    scored.reserve(nbrs.size());
    const float* node_vec = vector_at(node_idx);
    for (uint32_t nb : nbrs) {
        scored.push_back({dot_product(node_vec, vector_at(nb), dim_), nb});
    }
    std::sort(scored.begin(), scored.end(),
              [](const Item& a, const Item& b) { return a.score > b.score; });

    nbrs.clear();
    nbrs.reserve(max_degree);
    for (size_t i = 0; i < max_degree && i < scored.size(); ++i) {
        nbrs.push_back(scored[i].idx);
    }
}

void NSWIndex::add(VectorId id, const std::vector<float>& vector) {
    if (vector.size() != dim_) throw std::invalid_argument("vector dimension mismatch");

    std::vector<float> q(vector);
    normalize(q.data(), dim_);

    // IMPORTANT: search the graph as it exists *before* inserting the new
    // node, otherwise the new node (with an empty, still-unconnected edge
    // list) could pick itself as an entry point and short-circuit the
    // search with a spurious self-similarity of 1.0.
    std::vector<std::pair<uint32_t, float>> candidates;
    if (!ids_.empty()) {
        candidates = beam_search(q, ef_construction_);
    }

    uint32_t new_idx = static_cast<uint32_t>(ids_.size());
    ids_.push_back(id);
    size_t offset = data_.size();
    data_.resize(offset + dim_);
    std::memcpy(data_.data() + offset, q.data(), dim_ * sizeof(float));
    edges_.emplace_back();

    size_t connect_count = std::min(m_, candidates.size());
    for (size_t i = 0; i < connect_count; ++i) {
        uint32_t cand_idx = candidates[i].first;
        edges_[new_idx].push_back(cand_idx);
        edges_[cand_idx].push_back(new_idx);
        prune_edges(cand_idx);
    }
}

std::vector<SearchResult> NSWIndex::search(const std::vector<float>& query, size_t k,
                                            size_t ef) const {
    if (query.size() != dim_) throw std::invalid_argument("query dimension mismatch");
    ef = std::max(ef, k);
    std::vector<float> q(query);
    normalize(q.data(), dim_);

    auto candidates = beam_search(q, ef);
    size_t n = std::min(k, candidates.size());
    std::vector<SearchResult> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back({ids_[candidates[i].first], candidates[i].second});
    }
    return out;
}

std::vector<uint8_t> NSWIndex::serialize() const {
    std::vector<uint8_t> buf;
    auto push = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), b, b + n);
    };
    uint64_t dim64 = dim_, m64 = m_, efc64 = ef_construction_, count64 = ids_.size();
    push(&dim64, sizeof(dim64));
    push(&m64, sizeof(m64));
    push(&efc64, sizeof(efc64));
    push(&count64, sizeof(count64));
    push(ids_.data(), ids_.size() * sizeof(VectorId));
    push(data_.data(), data_.size() * sizeof(float));
    for (const auto& nbrs : edges_) {
        uint32_t degree = static_cast<uint32_t>(nbrs.size());
        push(&degree, sizeof(degree));
        push(nbrs.data(), nbrs.size() * sizeof(uint32_t));
    }
    return buf;
}

NSWIndex NSWIndex::deserialize(const std::vector<uint8_t>& blob) {
    size_t offset = 0;
    auto pull = [&](void* p, size_t n) {
        if (offset + n > blob.size()) throw std::runtime_error("corrupt NSWIndex blob");
        std::memcpy(p, blob.data() + offset, n);
        offset += n;
    };
    uint64_t dim64 = 0, m64 = 0, efc64 = 0, count64 = 0;
    pull(&dim64, sizeof(dim64));
    pull(&m64, sizeof(m64));
    pull(&efc64, sizeof(efc64));
    pull(&count64, sizeof(count64));

    NSWIndex idx(static_cast<size_t>(dim64), static_cast<size_t>(m64), static_cast<size_t>(efc64));
    size_t count = static_cast<size_t>(count64);
    idx.ids_.resize(count);
    pull(idx.ids_.data(), count * sizeof(VectorId));
    idx.data_.resize(count * static_cast<size_t>(dim64));
    pull(idx.data_.data(), idx.data_.size() * sizeof(float));

    idx.edges_.resize(count);
    for (size_t i = 0; i < count; ++i) {
        uint32_t degree = 0;
        pull(&degree, sizeof(degree));
        idx.edges_[i].resize(degree);
        pull(idx.edges_[i].data(), degree * sizeof(uint32_t));
    }
    return idx;
}

} // namespace cognivore
