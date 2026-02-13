#include "SimpleGraph.h"
#include <fstream>
#include <sstream>
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

    std::string SimpleGraph::GetKnowledgeContext(const std::vector<std::string>& entities) const {
        std::stringstream ss;
        for (const auto& entity : entities) {
            ss << GetKnowledgeContext(entity);
        }
        return ss.str();
    }

} // namespace NPCInference
