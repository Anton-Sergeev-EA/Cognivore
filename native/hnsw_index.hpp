#pragma once

#include "vector_index.hpp"

#include <shared_mutex>

namespace cognivore {

/// Hierarchical Navigable Small World graph over normalized float vectors.
class HNSWIndex {
public:
    explicit HNSWIndex(size_t dim, size_t m = 16, size_t ef_construction = 200,
                       uint32_t seed = 42);

    void add(VectorId id, const std::vector<float>& vector);
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k, size_t ef) const;

    size_t size() const;
    size_t dim() const noexcept { return dim_; }
    size_t max_level() const;

private:
    using NodeId = uint32_t;
    using Candidate = std::pair<float, NodeId>;
    using Layers = std::vector<std::vector<NodeId>>;

    size_t sample_level();
    NodeId descend(const float* query, NodeId entry, size_t layer) const;
    std::vector<Candidate> search_layer(const float* query, NodeId entry, size_t layer,
                                        size_t ef) const;
    std::vector<NodeId> select_neighbors(const std::vector<Candidate>& candidates,
                                         size_t limit) const;
    std::vector<NodeId> reverse_link(NodeId source, NodeId target, size_t layer) const;
    const float* vector_at(NodeId node) const noexcept {
        return data_.data() + static_cast<size_t>(node) * dim_;
    }

    size_t dim_;
    size_t m_;
    size_t ef_construction_;
    size_t max_level_ = 0;
    NodeId entry_ = 0;
    std::mt19937 rng_;
    mutable std::shared_mutex mutex_;
    std::vector<VectorId> ids_;
    std::vector<float> data_;
    std::vector<Layers> graph_;
};

} // namespace cognivore
