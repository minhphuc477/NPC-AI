#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <queue>
#include <set>
#include <mutex>

namespace NPCInference {

    struct GraphEdge {
        std::string relation;
        std::string target;
        float weight;
    };

    struct GraphPath {
        std::vector<std::string> nodes;
        std::vector<std::string> relations;
        float totalWeight;
    };

    class SimpleGraph {
    public:
        SimpleGraph() = default;
        ~SimpleGraph() = default;

        // Add a relationship: Subject -> Relation -> Target
        void AddRelation(const std::string& subject, const std::string& relation, const std::string& target, float weight = 1.0f);

        // Get all outgoing edges from a node
        std::vector<GraphEdge> GetRelations(const std::string& subject) const;

        // Find a path between two entities (BFS)
        // Returns empty path if no connection found within maxDepth
        GraphPath FindPath(const std::string& start, const std::string& end, int maxDepth = 3) const;

        // Check if node exists
        bool HasNode(const std::string& node) const;

        // Get a textual representation of direct facts about an entity
        std::string GetKnowledgeContext(const std::string& entity) const;
        std::string GetKnowledgeContext(const std::vector<std::string>& entities, int limit = 5) const;

        // Community Detection (Label Propagation)
        // Returns Map: CommunityID -> List of Member Nodes
        // Community Detection (Label Propagation)
        // Returns Map: CommunityID -> List of Member Nodes
        std::map<int, std::vector<std::string>> DetectCommunities();

        // Calculate PageRank for all nodes
        // Returns Map: Node -> Rank Score (0.0 - 1.0)
        std::map<std::string, float> CalculatePageRank(int max_iters = 20, float damping = 0.85f) const;

        // Serialization
        bool Save(const std::string& filepath) const;
        bool Load(const std::string& filepath);

    private:
        // Adjacency list: Node -> List of Edges
        std::map<std::string, std::vector<GraphEdge>> adjacencyList_;
        mutable std::recursive_mutex mutex_;
    };

} // namespace NPCInference
