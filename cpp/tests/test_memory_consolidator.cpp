#include <iostream>
#include "NPCInference.h"
#include "MemoryConsolidator.h"

int main() {
    std::cout << "Testing Memory Consolidator..." << std::endl;

    NPCInference::ModelLoader model_loader;
    NPCInference::Tokenizer tokenizer; // We need a real tokenizer for this to work
    
    // Construct directly
    NPCInference::MemoryConsolidator consolidator(&model_loader, &tokenizer);
    
    // Create dummy memories
    std::vector<NPCInference::UnconsolidatedMemory> memories;
    memories.push_back({"Hello there!", "Player", {}});
    memories.push_back({"General Kenobi! You are a bold one.", "NPC", {}});
    
    // We can't actually run inference without loading a model, but we can verify compilation/linking
    std::cout << "Memory Consolidator instantiated successfully." << std::endl;
    
    // If we had a mock ModelLoader, we could test prompt generation vs output.
    
    return 0;
}
