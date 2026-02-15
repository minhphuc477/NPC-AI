#include "TemporalMemorySystem.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace NPCInference;

/**
 * Demonstration of Temporal Memory System
 * 
 * Shows:
 * 1. Episodic memories with emotional context
 * 2. Memory decay over time
 * 3. Retrieval strengthening (spaced repetition)
 * 4. Semantic knowledge consolidation
 * 5. Persistence (save/load)
 */

void PrintSeparator() {
    std::cout << "\n" << std::string(60, '=') << "\n\n";
}

void SimulateTimePassage(int seconds) {
    std::cout << "[Simulating " << seconds << " seconds passing...]\n";
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

int main() {
    std::cout << "=== Temporal Memory System Demo ===\n";
    std::cout << "Demonstrating biologically-inspired memory with decay\n";
    PrintSeparator();
    
    // Create memory system
    TemporalMemorySystem memory;
    
    // Configure decay parameters
    memory.SetDecayRate(0.01f);  // Faster decay for demo
    memory.SetEmotionalBoost(3.0f);  // Strong emotional memories last longer
    
    // === Day 1: Meeting the Player ===
    std::cout << "DAY 1: First Encounter\n";
    std::cout << "------------------------\n";
    
    std::string ep1 = memory.AddEpisode(
        "Met a traveler named Alex at the market square. They asked about the old ruins.",
        0.3f,  // Slightly positive
        0.5f,  // Moderate arousal
        0.6f,  // Moderately important
        {"Alex", "Elara"},
        "Market Square"
    );
    std::cout << "✓ Added: First meeting with Alex\n";
    
    std::string ep2 = memory.AddEpisode(
        "Alex saved me from bandits! They fought bravely and asked for nothing in return.",
        0.9f,  // Very positive
        0.9f,  // High arousal (intense moment)
        0.95f, // Very important
        {"Alex", "Elara", "Bandits"},
        "Forest Road"
    );
    std::cout << "✓ Added: Alex saved Elara from bandits (EMOTIONAL)\n";
    
    std::string ep3 = memory.AddEpisode(
        "Sold Alex a health potion for 50 gold.",
        0.1f,  // Neutral-positive
        0.2f,  // Low arousal (routine)
        0.3f,  // Low importance
        {"Alex", "Elara"},
        "Market Square"
    );
    std::cout << "✓ Added: Routine transaction\n";
    
    PrintSeparator();
    
    // === Immediate Retrieval ===
    std::cout << "IMMEDIATE RECALL (Fresh Memories)\n";
    std::cout << "Query: 'Alex'\n";
    std::cout << "------------------------\n";
    
    auto memories = memory.RetrieveEpisodes("Alex", 10, 0.1f);
    for (size_t i = 0; i < memories.size(); i++) {
        std::cout << (i+1) << ". " << memories[i].description << "\n";
        std::cout << "   Strength: " << memories[i].current_strength 
                  << " | Importance: " << memories[i].importance
                  << " | Arousal: " << memories[i].emotional_arousal << "\n";
    }
    
    PrintSeparator();
    
    // === Time Passes ===
    std::cout << "⏰ 5 SECONDS LATER...\n";
    SimulateTimePassage(5);
    
    std::cout << "\nRECALL AFTER TIME (Memory Decay)\n";
    std::cout << "Query: 'Alex'\n";
    std::cout << "------------------------\n";
    
    memories = memory.RetrieveEpisodes("Alex", 10, 0.1f);
    for (size_t i = 0; i < memories.size(); i++) {
        std::cout << (i+1) << ". " << memories[i].description << "\n";
        std::cout << "   Strength: " << memories[i].current_strength 
                  << " (decayed from 1.0)\n";
    }
    
    std::cout << "\n💡 Notice: Emotional memory (bandit rescue) decays SLOWER than routine transaction!\n";
    
    PrintSeparator();
    
    // === More Time Passes ===
    std::cout << "⏰ 10 MORE SECONDS LATER...\n";
    SimulateTimePassage(10);
    
    std::cout << "\nRECALL AFTER MORE TIME\n";
    std::cout << "Query: 'bandits'\n";
    std::cout << "------------------------\n";
    
    memories = memory.RetrieveEpisodes("bandits", 10, 0.05f);
    for (const auto& mem : memories) {
        std::cout << "• " << mem.description << "\n";
        std::cout << "  Strength: " << mem.current_strength << "\n";
    }
    
    PrintSeparator();
    
    // === Add More Episodes for Consolidation ===
    std::cout << "DAY 2-5: More Interactions\n";
    std::cout << "------------------------\n";
    
    memory.AddEpisode(
        "Alex returned to buy another potion.",
        0.2f, 0.3f, 0.4f,
        {"Alex", "Elara"},
        "Market Square"
    );
    std::cout << "✓ Day 2: Alex bought another potion\n";
    
    memory.AddEpisode(
        "Alex asked about my family. I told them about my late husband.",
        0.5f, 0.6f, 0.7f,
        {"Alex", "Elara"},
        "Market Square"
    );
    std::cout << "✓ Day 3: Personal conversation\n";
    
    memory.AddEpisode(
        "Alex brought me flowers. How thoughtful!",
        0.8f, 0.7f, 0.8f,
        {"Alex", "Elara"},
        "Market Square"
    );
    std::cout << "✓ Day 4: Alex brought flowers\n";
    
    PrintSeparator();
    
    // === Semantic Knowledge ===
    std::cout << "SEMANTIC KNOWLEDGE (General Beliefs)\n";
    std::cout << "------------------------\n";
    
    memory.AddSemanticKnowledge(
        "Alex is trustworthy and kind-hearted.",
        0.9f,  // High confidence
        ep2    // Based on bandit rescue
    );
    std::cout << "✓ Formed belief: Alex is trustworthy (from bandit rescue)\n";
    
    memory.AddSemanticKnowledge(
        "Alex is a regular customer.",
        0.7f
    );
    std::cout << "✓ Formed belief: Alex is a regular customer\n";
    
    auto semantic = memory.RetrieveSemanticKnowledge("Alex", 5);
    std::cout << "\nGeneral knowledge about Alex:\n";
    for (const auto& know : semantic) {
        std::cout << "• " << know.knowledge 
                  << " (confidence: " << know.confidence << ")\n";
    }
    
    PrintSeparator();
    
    // === Memory Consolidation ===
    std::cout << "MEMORY CONSOLIDATION (Sleep Cycle)\n";
    std::cout << "------------------------\n";
    std::cout << "Simulating overnight consolidation...\n\n";
    
    // Mock LLM for pattern extraction
    auto mock_llm = [](const std::string& prompt) -> std::string {
        if (prompt.find("Alex") != std::string::npos) {
            return "Alex is becoming a trusted friend and regular customer.";
        }
        return "General pattern extracted.";
    };
    
    memory.ConsolidateMemories(mock_llm);
    std::cout << "✓ Consolidation complete\n\n";
    
    semantic = memory.RetrieveSemanticKnowledge("Alex", 5);
    std::cout << "Updated semantic knowledge:\n";
    for (const auto& know : semantic) {
        std::cout << "• " << know.knowledge 
                  << " (confidence: " << know.confidence << ")\n";
    }
    
    PrintSeparator();
    
    // === Statistics ===
    std::cout << "MEMORY STATISTICS\n";
    std::cout << "------------------------\n";
    
    auto stats = memory.GetStats();
    std::cout << "Total Episodes: " << stats.total_episodes << "\n";
    std::cout << "Active Episodes (strength > 0.1): " << stats.active_episodes << "\n";
    std::cout << "Total Semantic Knowledge: " << stats.total_semantic << "\n";
    std::cout << "Average Episode Strength: " << stats.avg_episode_strength << "\n";
    
    PrintSeparator();
    
    // === Persistence ===
    std::cout << "PERSISTENCE TEST\n";
    std::cout << "------------------------\n";
    
    std::string save_path = "elara_memories.json";
    if (memory.Save(save_path)) {
        std::cout << "✓ Saved memories to " << save_path << "\n";
    }
    
    // Create new memory system and load
    TemporalMemorySystem loaded_memory;
    if (loaded_memory.Load(save_path)) {
        std::cout << "✓ Loaded memories from " << save_path << "\n\n";
        
        auto loaded_stats = loaded_memory.GetStats();
        std::cout << "Loaded " << loaded_stats.total_episodes << " episodes\n";
        std::cout << "Loaded " << loaded_stats.total_semantic << " semantic memories\n";
    }
    
    PrintSeparator();
    
    // === Final Demonstration ===
    std::cout << "FINAL DEMONSTRATION: Entity-Based Retrieval\n";
    std::cout << "------------------------\n";
    std::cout << "All memories involving Alex:\n\n";
    
    auto alex_memories = memory.GetEpisodesWithEntity("Alex", 20);
    for (size_t i = 0; i < alex_memories.size(); i++) {
        std::cout << (i+1) << ". " << alex_memories[i].description << "\n";
        std::cout << "   @ " << alex_memories[i].location 
                  << " | Strength: " << alex_memories[i].current_strength << "\n";
    }
    
    PrintSeparator();
    
    std::cout << "✅ DEMO COMPLETE\n\n";
    std::cout << "Key Takeaways:\n";
    std::cout << "1. Emotional memories decay slower than routine ones\n";
    std::cout << "2. Important events are remembered longer\n";
    std::cout << "3. Retrieval strengthens memories (spaced repetition)\n";
    std::cout << "4. Episodes consolidate into semantic knowledge\n";
    std::cout << "5. Full persistence for long-term NPC memory\n\n";
    std::cout << "This solves the unsolved problem of NPCs forgetting context!\n";
    
    return 0;
}
