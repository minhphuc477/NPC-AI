#include "NPCInference.h"
#include "OllamaClient.h"
#include <iostream>
#include <string>

using namespace NPCInference;

int main() {
    std::cout << "=== NPC Chat Interface ===\n";
    std::cout << "Initializing NPC Engine...\n";
    
    // Initialize Ollama client with fine-tuned NPC model
    auto ollama = std::make_unique<OllamaClient>("elara-npc");
    
    if (!ollama->IsReady()) {
        std::cerr << "\nError: Ollama is not running!\n";
        std::cerr << "Please start Ollama with: ollama serve\n";
        std::cerr << "Then make sure phi3:mini is available: ollama pull phi3:mini\n";
        return 1;
    }
    
    std::cout << "✓ Ollama connected successfully!\n";
    
    // Initialize NPC engine (for memory, emotions, social systems)
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "F:/NPC AI/models/phi3_onnx";  // Not used for LLM anymore
    config.enable_rag = true;
    config.enable_graph = true;
    config.enable_reflection = false;
    config.enable_planner = false;
    config.enable_truth_guard = false;
    
    auto engine = std::make_unique<NPCInferenceEngine>();
    
    if (!engine->Initialize(config)) {
        std::cerr << "Warning: NPC engine initialization had issues, but continuing...\n";
    }
    
    std::cout << "\n=== Chat with Elara the Merchant ===\n";
    std::cout << "Commands:\n";
    std::cout << "  quit - Exit the chat\n";
    std::cout << "  help - Show this help\n";
    std::cout << "  learn <text> - Teach Elara a new fact\n";
    std::cout << "\nStart chatting!\n\n";
    
    std::string session_id = engine->StartConversation("Elara", "Player");
    
    // Add some initial knowledge
    engine->Learn("Elara sells health potions for 50 gold.");
    engine->Learn("Elara is located in the market square.");
    engine->Learn("Elara is friendly to travelers.");
    
    // System prompt for Elara
    std::string system_prompt = 
        "You are Elara, a friendly merchant NPC in a fantasy RPG game. "
        "You sell health potions and various supplies. You're located in the market square. "
        "Keep responses brief (1-2 sentences) and stay in character. "
        "Be helpful and friendly to the player.";
    
    while (true) {
        std::cout << "You: ";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input.empty()) continue;
        
        if (user_input == "quit") {
            std::cout << "\nElara: Safe travels, friend!\n";
            break;
        }
        
        if (user_input == "help") {
            std::cout << "\nCommands:\n";
            std::cout << "  quit - Exit the chat\n";
            std::cout << "  help - Show this help\n";
            std::cout << "  learn <text> - Teach Elara a new fact\n\n";
            continue;
        }
        
        if (user_input.substr(0, 6) == "learn ") {
            std::string fact = user_input.substr(6);
            engine->Learn(fact);
            std::cout << "Elara: *nods* I'll remember that.\n\n";
            continue;
        }
        
        // Build context from advanced NPC systems
        auto context = engine->BuildAdvancedContext("Elara", user_input);
        
        // Build prompt with context
        // Build prompt matching fine-tuning format: [CONTEXT] {json} [PLAYER] {input}
        // The model was trained to parse this specific structure
        std::string full_prompt = "[CONTEXT]\n" + context.dump() + "\n\n[PLAYER] " + user_input;
        
        // Generate response using Ollama
        std::string response = ollama->Generate(full_prompt, 100, 0.7f);
        
        std::cout << "Elara: " << response << "\n\n";
        
        // Store interaction in memory
        engine->GetTemporalMemory()->AddEpisode(
            "Player said: " + user_input,
            0.5f,  // neutral valence
            0.3f,  // low arousal
            0.4f   // moderate importance
        );
    }
    
    return 0;
}
