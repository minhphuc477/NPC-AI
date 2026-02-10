#pragma once

#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace NPCBehavior {

    // Status of a node execution
    enum class Status {
        SUCCESS,
        FAILURE,
        RUNNING
    };

    // Blackboard for sharing data between nodes
    // Using json for flexibility to match Python implementation
    using Blackboard = json;

    // Abstract base class for all BT nodes
    class Node {
    public:
        virtual ~Node() = default;
        virtual Status tick(Blackboard& blackboard) = 0;
    };

    // Composite node (has children)
    class Composite : public Node {
    protected:
        std::vector<std::shared_ptr<Node>> children;
    public:
        Composite(const std::vector<std::shared_ptr<Node>>& children) : children(children) {}
        void addChild(std::shared_ptr<Node> child) {
            children.push_back(child);
        }
    };

    // Selector: Succeeds if ANY child succeeds
    class Selector : public Composite {
    public:
        using Composite::Composite;
        Status tick(Blackboard& blackboard) override {
            for (auto& child : children) {
                Status status = child->tick(blackboard);
                if (status != Status::FAILURE) {
                    return status;
                }
            }
            return Status::FAILURE;
        }
    };

    // Sequence: Succeeds only if ALL children succeed
    class Sequence : public Composite {
    public:
        using Composite::Composite;
        Status tick(Blackboard& blackboard) override {
            for (auto& child : children) {
                Status status = child->tick(blackboard);
                if (status == Status::FAILURE || status == Status::RUNNING) {
                    return status;
                }
            }
            return Status::SUCCESS;
        }
    };

    // Condition: Checks a predicate
    class Condition : public Node {
        std::function<bool(const Blackboard&)> predicate;
    public:
        Condition(std::function<bool(const Blackboard&)> predicate) : predicate(predicate) {}
        Status tick(Blackboard& blackboard) override {
            if (predicate(blackboard)) {
                return Status::SUCCESS;
            }
            return Status::FAILURE;
        }
    };

    // Action: Performs a task
    class Action : public Node {
        std::function<Status(Blackboard&)> action;
    public:
        Action(std::function<Status(Blackboard&)> action) : action(action) {}
        Status tick(Blackboard& blackboard) override {
            return action(blackboard);
        }
    };

    // Factory to create the specific NPC behavior tree
    std::shared_ptr<Node> CreateNPCBehaviorTree();

} // namespace NPCBehavior
