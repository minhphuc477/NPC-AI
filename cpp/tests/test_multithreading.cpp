#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <atomic>
#include <cassert>
#include "SocialFabricNetwork.h"
#include "SimpleGraph.h"
#include "AmbientAwarenessSystem.h"
#include "PlayerBehaviorModeling.h"
#include "TemporalMemorySystem.h"
#include "NPCLogger.h"

using namespace NPCInference;

void StressTestSocialFabric(SocialFabricNetwork& network, int num_threads, int operations_per_thread) {
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&network, &start_flag, operations_per_thread, t]() {
            while (!start_flag) std::this_thread::yield();

            std::mt19937 gen(1337 + t);
            std::uniform_int_distribution<> npc_dist(0, 50);

            for (int i = 0; i < operations_per_thread; ++i) {
                std::string npc_a = "NPC_" + std::to_string(npc_dist(gen));
                std::string npc_b = "NPC_" + std::to_string(npc_dist(gen));
                if (npc_a == npc_b) continue;

                if (i % 2 == 0) {
                    network.UpdateRelationship(npc_a, npc_b, 0.1f, 0.05f, 0.02f, "exp_" + std::to_string(i));
                } else {
                    auto rel = network.GetRelationship(npc_a, npc_b);
                    (void)rel;
                }
            }
        });
    }

    NPCLogger::Info("Starting SocialFabricNetwork Multithreading Stress Test...");
    auto start_time = std::chrono::high_resolution_clock::now();
    start_flag = true;

    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    NPCLogger::Info("SocialFabricNetwork Stress Test Completed in " + std::to_string(duration.count()) + "ms.");
}

void StressTestSimpleGraph(SimpleGraph& graph, int num_threads, int operations_per_thread) {
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&graph, &start_flag, operations_per_thread, t]() {
            while (!start_flag) std::this_thread::yield();

            std::mt19937 gen(2337 + t);
            std::uniform_int_distribution<> node_dist(0, 30);

            for (int i = 0; i < operations_per_thread; ++i) {
                std::string node_a = "Concept_" + std::to_string(node_dist(gen));
                std::string node_b = "Concept_" + std::to_string(node_dist(gen));
                
                if (i % 2 == 0) {
                    graph.AddRelation(node_a, "related_to", node_b, 0.5f);
                } else {
                    auto path = graph.FindPath(node_a, node_b, 3);
                    (void)path;
                }
            }
        });
    }

    NPCLogger::Info("Starting SimpleGraph Multithreading Stress Test...");
    auto start_time = std::chrono::high_resolution_clock::now();
    start_flag = true;

    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    NPCLogger::Info("SimpleGraph Stress Test Completed in " + std::to_string(duration.count()) + "ms.");
}

void StressTestAmbientAwareness(AmbientAwarenessSystem& system, int num_threads, int operations_per_thread) {
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&system, &start_flag, operations_per_thread, t]() {
            while (!start_flag) std::this_thread::yield();

            std::mt19937 gen(3337 + t);
            std::uniform_int_distribution<> type_dist(0, 5);
            std::vector<std::string> types = {"combat", "noise", "blood", "movement", "theft", "fire"};

            for (int i = 0; i < operations_per_thread; ++i) {
                std::string type = types[type_dist(gen)];
                
                if (i % 3 == 0) {
                    system.ObserveEvent(type, "Concurrent Observation " + std::to_string(i), {"NPC_X"}, "Area_1");
                } else if (i % 3 == 1) {
                    system.RecordEvidence(type, "Concurrent Evidence " + std::to_string(i), "Area_1", 0.9f);
                } else {
                    auto aware = system.IsAwareOf(type);
                    (void)aware;
                }
            }
        });
    }

    NPCLogger::Info("Starting AmbientAwarenessSystem Multithreading Stress Test...");
    auto start_time = std::chrono::high_resolution_clock::now();
    start_flag = true;

    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    NPCLogger::Info("AmbientAwarenessSystem Stress Test Completed in " + std::to_string(duration.count()) + "ms.");
}

void StressTestBehaviorModeling(PlayerBehaviorModeling& modeling, int num_threads, int operations_per_thread) {
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&modeling, &start_flag, operations_per_thread, t]() {
            while (!start_flag) std::this_thread::yield();

            std::mt19937 gen(4337 + t);
            std::vector<std::string> actions = {"attack", "defend", "negotiate", "scout", "bribe"};
            std::uniform_int_distribution<> act_dist(0, 4);

            for (int i = 0; i < operations_per_thread; ++i) {
                std::string action = actions[act_dist(gen)];
                
                if (i % 2 == 0) {
                    modeling.RecordAction(action, "Target", "Combat", true, 0.5f);
                } else {
                    auto strategy = modeling.SuggestCounterStrategy("Defeat Player");
                    (void)strategy;
                }
            }
        });
    }

    NPCLogger::Info("Starting PlayerBehaviorModeling Multithreading Stress Test...");
    auto start_time = std::chrono::high_resolution_clock::now();
    start_flag = true;

    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    NPCLogger::Info("PlayerBehaviorModeling Stress Test Completed in " + std::to_string(duration.count()) + "ms.");
}

int main() {
    SocialFabricNetwork social;
    StressTestSocialFabric(social, 8, 1000);
    
    SimpleGraph graph;
    StressTestSimpleGraph(graph, 8, 1000);
    
    AmbientAwarenessSystem ambient;
    StressTestAmbientAwareness(ambient, 8, 1000);
    
    PlayerBehaviorModeling modeling;
    StressTestBehaviorModeling(modeling, 8, 1000);
    
    NPCLogger::Info("All Multi-threaded Hardening Tests PASSED without crashes/deadlocks.");
    
    return 0;
}
