#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <atomic>
#include <cassert>
#include "SocialFabricNetwork.h"
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

int main() {
    SocialFabricNetwork network;
    StressTestSocialFabric(network, 8, 1000);
    
    // Final check
    auto stats = network.GetStats();
    NPCLogger::Info("Final Social Stats: Relationships=" + std::to_string(stats.total_relationships));
    
    return 0;
}
