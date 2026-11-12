#include "hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <tuple>
#include <queue>
#include <stdexcept>

namespace cognivore {
namespace {
std::vector<float> unit_vector(const std::vector<float>& input, size_t dim) {
    if (input.size() != dim) throw std::invalid_argument("vector dimension mismatch");
    double squared = 0.0;
    for (double value : input) {
        if (!std::isfinite(value)) throw std::invalid_argument("vector must be finite");
        squared += value * value;
    }
    std::vector<float> output(input);
    if (squared > 0.0) {
        const double norm = std::sqrt(squared);
        for (float& value : output) value = static_cast<float>(value / norm);
    }
    return output;
}
} // namespace

HNSWIndex::HNSWIndex(size_t dim, size_t m, size_t ef_construction, uint32_t seed)
    : dim_(dim), m_(m), ef_construction_(ef_construction), rng_(seed) {
    if (dim == 0) throw std::invalid_argument("dimension must be positive");
    if (m < 2 || m > 1024) throw std::invalid_argument("m must be between 2 and 1024");
    if (ef_construction < m) throw std::invalid_argument("ef_construction must be at least m");
}

size_t HNSWIndex::sample_level() {
    const double uniform = (static_cast<double>(rng_()) + 1.0) /
                           (static_cast<double>(std::mt19937::max()) + 1.0);
    return std::min<size_t>(32, static_cast<size_t>(-std::log(uniform) / std::log(m_)));
}

HNSWIndex::NodeId HNSWIndex::descend(const float* query, NodeId entry, size_t layer) const {
    float score = dot_product(query, vector_at(entry), dim_);
    while (true) {
        NodeId next = entry;
        for (NodeId neighbor : graph_[entry][layer]) {
            float candidate = dot_product(query, vector_at(neighbor), dim_);
            if (candidate > score) {
                score = candidate;
                next = neighbor;
            }
        }
        if (next == entry) return entry;
        entry = next;
    }
}

std::vector<HNSWIndex::Candidate> HNSWIndex::search_layer(const float* query, NodeId entry,
                                                         size_t layer, size_t ef) const {
    std::priority_queue<Candidate> frontier;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> best;
    std::unordered_set<NodeId> visited{entry};
    const Candidate start{dot_product(query, vector_at(entry), dim_), entry};
    frontier.push(start);
    best.push(start);
    while (!frontier.empty()) {
        const auto current = frontier.top();
        frontier.pop();
        if (best.size() == ef && current.first < best.top().first) break;
        for (NodeId neighbor : graph_[current.second][layer]) {
            if (!visited.insert(neighbor).second) continue;
            const Candidate candidate{dot_product(query, vector_at(neighbor), dim_), neighbor};
            if (best.size() < ef || candidate.first > best.top().first) {
                frontier.push(candidate);
                best.push(candidate);
                if (best.size() > ef) best.pop();
            }
        }
    }
    std::vector<Candidate> sorted(best.size());
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        *it = best.top();
        best.pop();
    }
    return sorted;
}

std::vector<HNSWIndex::NodeId>
HNSWIndex::select_neighbors(const std::vector<Candidate>& candidates, size_t limit) const {
    std::vector<NodeId> selected;
    std::vector<NodeId> deferred;
    selected.reserve(std::min(candidates.size(), limit));
    for (const auto& [similarity, candidate] : candidates) {
        if (selected.size() == limit) break;
        const bool redundant = std::any_of(selected.begin(), selected.end(), [&](NodeId neighbor) {
            return dot_product(vector_at(candidate), vector_at(neighbor), dim_) > similarity;
        });
        if (redundant) {
            deferred.push_back(candidate);
        } else {
            selected.push_back(candidate);
        }
    }
    const size_t fill = std::min(limit - selected.size(), deferred.size());
    selected.insert(selected.end(), deferred.begin(), deferred.begin() + fill);
    return selected;
}

std::vector<HNSWIndex::NodeId> HNSWIndex::reverse_link(NodeId source, NodeId target,
                                                       size_t layer) const {
    auto neighbors = graph_[source][layer];
    neighbors.push_back(target);
    const size_t limit = layer == 0 ? 2 * m_ : m_;
    if (neighbors.size() <= limit) return neighbors;

    std::vector<Candidate> candidates;
    candidates.reserve(neighbors.size());
    for (NodeId neighbor : neighbors) {
        candidates.emplace_back(dot_product(vector_at(source), vector_at(neighbor), dim_), neighbor);
    }
    std::sort(candidates.begin(), candidates.end(), std::greater<Candidate>{});
    return select_neighbors(candidates, limit);
}

void HNSWIndex::add(VectorId id, const std::vector<float>& vector) {
    auto query = unit_vector(vector, dim_);
    const std::unique_lock lock(mutex_);
    if (ids_.size() == std::numeric_limits<NodeId>::max()) {
        throw std::length_error("HNSW node limit reached");
    }
    const auto node = static_cast<NodeId>(ids_.size());
    const size_t level = sample_level();

    Layers edges(level + 1);
    if (node != 0) {
        NodeId entry = entry_;
        for (size_t layer = max_level_; layer > level; --layer) {
            entry = descend(query.data(), entry, layer);
        }
        for (size_t layers = std::min(level, max_level_) + 1; layers > 0; --layers) {
            const size_t layer = layers - 1;
            const auto candidates = search_layer(query.data(), entry, layer, ef_construction_);
            edges[layer] = select_neighbors(candidates, m_);
            entry = candidates.front().second;
        }
    }

    // Prepare every reverse link before committing adjacency changes.
    using ReverseLink = std::tuple<NodeId, size_t, std::vector<NodeId>>;
    std::vector<ReverseLink> updates;
    const size_t old_data_size = data_.size();
    data_.insert(data_.end(), query.begin(), query.end());
    try {
        for (size_t layer = 0; layer <= level; ++layer) {
            for (NodeId neighbor : edges[layer]) {
                updates.emplace_back(neighbor, layer, reverse_link(neighbor, node, layer));
            }
        }
        ids_.push_back(id);
        graph_.push_back(std::move(edges));
    } catch (...) {
        data_.resize(old_data_size);
        if (ids_.size() > node) ids_.pop_back();
        throw;
    }
    for (auto& [neighbor, layer, links] : updates) {
        graph_[neighbor][layer] = std::move(links);
    }
    if (node == 0 || level > max_level_) {
        entry_ = node;
        max_level_ = level;
    }
}

std::vector<SearchResult> HNSWIndex::search(const std::vector<float>& query, size_t k,
                                           size_t ef) const {
    const auto normalized = unit_vector(query, dim_);
    const std::shared_lock lock(mutex_);
    if (ids_.empty() || k == 0) return {};
    NodeId entry = entry_;
    for (size_t layer = max_level_; layer > 0; --layer) {
        entry = descend(normalized.data(), entry, layer);
    }

    const auto candidates =
        search_layer(normalized.data(), entry, 0, std::min(ids_.size(), std::max(k, ef)));
    const size_t count = std::min(k, candidates.size());
    std::vector<SearchResult> results;
    results.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto [score, node] = candidates[i];
        results.push_back({ids_[node], score});
    }
    return results;
}


size_t HNSWIndex::size() const {
    const std::shared_lock lock(mutex_);
    return ids_.size();
}

size_t HNSWIndex::max_level() const {
    const std::shared_lock lock(mutex_);
    return max_level_;
}

} // namespace cognivore
