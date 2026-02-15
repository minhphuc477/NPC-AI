#include "SimpleGraph.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <random>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace NPCInference {

    void SimpleGraph::AddRelation(const std::string& subject, const std::string& relation, const std::string& target, float weight) {
        // Check if edge already exists
        auto& edges = adjacencyList_[subject];
        for (auto& edge : edges) {
            if (edge.relation == relation && edge.target == target) {
                // Update weight if exists
                edge.weight = weight;
                return;
            }
        }
        // Add new edge
        edges.push_back({relation, target, weight});
    }

    std::vector<GraphEdge> SimpleGraph::GetRelations(const std::string& subject) const {
        if (adjacencyList_.count(subject)) {
            return adjacencyList_.at(subject);
        }
        return {};
    }

    GraphPath SimpleGraph::FindPath(const std::string& start, const std::string& end, int maxDepth) const {
        if (!adjacencyList_.count(start)) return {};

        std::queue<std::pair<std::string, GraphPath>> q;
        std::set<std::string> visited;

        GraphPath initialPath;
        initialPath.nodes.push_back(start);
        initialPath.totalWeight = 0;

        q.push({start, initialPath});
        visited.insert(start);

        while (!q.empty()) {
            auto [current, path] = q.front();
            q.pop();

            if (current == end) {
                return path;
            }

            if (path.nodes.size() > maxDepth + 1) continue;

            if (adjacencyList_.count(current)) {
                for (const auto& edge : adjacencyList_.at(current)) {
                    if (visited.find(edge.target) == visited.end()) {
                        GraphPath newPath = path;
                        newPath.nodes.push_back(edge.target);
                        newPath.relations.push_back(edge.relation);
                        newPath.totalWeight += edge.weight;

                        if (edge.target == end) return newPath;

                        visited.insert(edge.target);
                        q.push({edge.target, newPath});
                    }
                }
            }
        }

        return {}; // No path found
    }

    std::string SimpleGraph::GetKnowledgeContext(const std::string& entity) const {
        std::stringstream ss;
        if (adjacencyList_.count(entity)) {
            const auto& edges = adjacencyList_.at(entity);
            for (const auto& edge : edges) {
                ss << entity << " " << edge.relation << " " << edge.target << ".\n";
            }
        }
        return ss.str();
    }

    bool SimpleGraph::Save(const std::string& filepath) const {
        json j;
        for (const auto& [node, edges] : adjacencyList_) {
            json edgeList = json::array();
            for (const auto& edge : edges) {
                edgeList.push_back({
                    {"r", edge.relation},
                    {"t", edge.target},
                    {"w", edge.weight}
                });
            }
            j[node] = edgeList;
        }

        try {
            std::ofstream f(filepath);
            f << j.dump(4);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool SimpleGraph::Load(const std::string& filepath) {
        try {
            std::ifstream f(filepath);
            if (!f.is_open()) return false;
            json j;
            f >> j;

            adjacencyList_.clear();
            for (const auto& el : j.items()) {
                std::string node = el.key();
                std::vector<GraphEdge> edges;
                for (const auto& edgeJson : el.value()) {
                    edges.push_back({
                        edgeJson["r"],
                        edgeJson["t"],
                        edgeJson.value("w", 1.0f)
                    });
                }
                adjacencyList_[node] = edges;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    bool SimpleGraph::HasNode(const std::string& node) const {
        return adjacencyList_.count(node) > 0;
    }

    std::string SimpleGraph::GetKnowledgeContext(const std::vector<std::string>& entities, int limit) const {
        // 1. Calculate Importance (PageRank)
        auto ranks = CalculatePageRank(10); // Quick iteration

        // 2. Sort entities by rank
        std::vector<std::pair<std::string, float>> sorted_entities;
        for (const auto& entity : entities) {
            float score = ranks.count(entity) ? ranks.at(entity) : 0.0f;
            sorted_entities.push_back({entity, score});
        }
        
        std::sort(sorted_entities.begin(), sorted_entities.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // 3. Select top K and build context
        std::stringstream ss;
        int count = 0;
        for (const auto& [entity, score] : sorted_entities) {
            if (count >= limit) break;
            ss << GetKnowledgeContext(entity); // Get edges for this important node
            count++;
        }
        return ss.str();
    }

    std::map<int, std::vector<std::string>> SimpleGraph::DetectCommunities() {
        // Label Propagation Algorithm (LPA)
        std::map<std::string, int> labels;
        std::vector<std::string> nodes;
        int next_label = 0;

        // 1. Initialize unique labels
        for (const auto& [node, _] : adjacencyList_) {
            labels[node] = next_label++;
            nodes.push_back(node);
        }

        if (nodes.empty()) return {};

        bool changed = true;
        int max_iters = 10;
        
        // 2. Iterate
        std::random_device rd;
        std::mt19937 g(rd());
        
        for (int i = 0; i < max_iters && changed; ++i) {
            changed = false;
            // Shuffle to prevent oscillation
            std::shuffle(nodes.begin(), nodes.end(), g); 

            for (const auto& node : nodes) {
                std::map<int, float> label_weights;
                
                // Collect neighbor labels
                if (adjacencyList_.count(node)) {
                    for (const auto& edge : adjacencyList_.at(node)) {
                        if (labels.count(edge.target)) {
                            label_weights[labels[edge.target]] += edge.weight;
                        }
                    }
                }

                if (label_weights.empty()) continue;

                // Find max weight label
                int best_label = labels[node];
                float max_w = -1.0f;
                for (const auto& [l, w] : label_weights) {
                    if (w > max_w) {
                        max_w = w;
                        best_label = l;
                    }
                }

                if (best_label != labels[node]) {
                    labels[node] = best_label;
                    changed = true;
                }
            }
        }

        // 3. Group by label
        std::map<int, std::vector<std::string>> communities;
        for (const auto& [node, label] : labels) {
            communities[label].push_back(node);
        }
        
        return communities;
    }

    std::map<std::string, float> SimpleGraph::CalculatePageRank(int max_iters, float damping) const {
        std::map<std::string, float> ranks;
        std::set<std::string> nodes;
        
        // 1. Collect all nodes
        for (const auto& [node, edges] : adjacencyList_) {
            nodes.insert(node);
            for (const auto& edge : edges) nodes.insert(edge.target);
        }
        
        size_t N = nodes.size();
        if (N == 0) return {};
        
        // 2. Initialize ranks
        float initial_rank = 1.0f / N;
        for (const auto& node : nodes) ranks[node] = initial_rank;
        
        // 3. Iteration
        for (int i = 0; i < max_iters; ++i) {
            std::map<std::string, float> new_ranks;
            float sink_rank = 0.0f;
            
            // Handle sink nodes (no outgoing edges)
            for (const auto& node : nodes) {
                if (adjacencyList_.find(node) == adjacencyList_.end() || adjacencyList_.at(node).empty()) {
                    sink_rank += ranks[node];
                }
            }
            
            for (const auto& node : nodes) {
                float rank_sum = 0.0f;
                
                // Find incoming edges (inefficient O(E), but valid for SimpleGraph)
                // Optimization: Pre-calculate reverse graph if performance needed
                for (const auto& [source, edges] : adjacencyList_) {
                    for (const auto& edge : edges) {
                        if (edge.target == node) {
                            rank_sum += ranks[source] / edges.size();
                        }
                    }
                }
                
                new_ranks[node] = (1.0f - damping) / N + damping * (rank_sum + sink_rank / N);
            }
            ranks = new_ranks;
        }
        
        return ranks;
    }

} // namespace NPCInference
