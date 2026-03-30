#include "BehaviorTree.h"
#include <iostream>

namespace NPCBehavior {

    // --- Conditions ---

    bool IsHpLow(const Blackboard& bb) {
        return bb.value("hp", 100) < 30;
    }

    bool IsPlayerNearby(const Blackboard& bb) {
        return bb.value("is_player_nearby", false);
    }

    bool IsPlayerTalking(const Blackboard& bb) {
        return bb.value("is_player_talking", false);
    }

    bool IsCombat(const Blackboard& bb) {
        return bb.value("is_combat", false);
    }

    // --- Actions ---

    Status ActionFlee(Blackboard& bb) {
        bb["current_action"] = "Fleeing";
        // In a real game, this would trigger navigation logic
        return Status::SUCCESS;
    }

    Status ActionAttack(Blackboard& bb) {
        bb["current_action"] = "Attacking";
        return Status::SUCCESS;
    }

    Status ActionDialogue(Blackboard& bb) {
        bb["current_action"] = "Talking";
        return Status::SUCCESS;
    }

    Status ActionPatrol(Blackboard& bb) {
        bb["current_action"] = "Patrolling";
        return Status::SUCCESS;
    }

    Status ActionIdle(Blackboard& bb) {
        bb["current_action"] = "Idle";
        return Status::SUCCESS;
    }

    // --- Tree Construction ---

    std::shared_ptr<Node> CreateNPCBehaviorTree() {
        // Combat Branch
        auto combatSeq = std::make_shared<Sequence>(std::vector<std::shared_ptr<Node>>{
            std::make_shared<Condition>(IsCombat),
            std::make_shared<Selector>(std::vector<std::shared_ptr<Node>>{
                std::make_shared<Sequence>(std::vector<std::shared_ptr<Node>>{
                    std::make_shared<Condition>(IsHpLow),
                    std::make_shared<Action>(ActionFlee)
                }),
                std::make_shared<Action>(ActionAttack)
            })
        });

        // Social Branch
        auto socialSeq = std::make_shared<Sequence>(std::vector<std::shared_ptr<Node>>{
            std::make_shared<Condition>(IsPlayerNearby),
            std::make_shared<Selector>(std::vector<std::shared_ptr<Node>>{
                std::make_shared<Sequence>(std::vector<std::shared_ptr<Node>>{
                    std::make_shared<Condition>(IsPlayerTalking),
                    std::make_shared<Action>(ActionDialogue)
                }),
                std::make_shared<Action>(ActionPatrol)
            })
        });

        // Root Selector
        auto root = std::make_shared<Selector>(std::vector<std::shared_ptr<Node>>{
            combatSeq,
            socialSeq,
            std::make_shared<Action>(ActionIdle)
        });

        return root;
    }

} // namespace NPCBehavior
