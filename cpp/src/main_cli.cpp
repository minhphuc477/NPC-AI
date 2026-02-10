// main_cli.cpp - Standalone CLI matching Python npc_cli.py interface

#include "NPCInference.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <sstream>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    // Initialize inference engine
    NPCInference::NPCInferenceEngine engine;
    
    std::cerr << "STATUS: Loading model..." << std::endl;
    
    // Load model (path should be passed as argument or environment variable)
    std::string model_path = argc > 1 ? argv[1] : "model.onnx";
    std::string adapter_path = argc > 2 ? argv[2] : "";
    
    if (!engine.LoadModel(model_path, adapter_path)) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        return 1;
    }
    
    std::cerr << "STATUS: Model loaded successfully!" << std::endl;
    std::cout << "READY" << std::endl; // Signal to parent process
    std::cout.flush();
    
    // Main loop - read JSON from stdin, write JSON to stdout
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        
        try {
            // Parse JSON request
            json request = json::parse(line);
            
            // Check for exit command
            if (request.contains("command") && request["command"] == "exit") {
                std::cerr << "Received exit command, shutting down..." << std::endl;
                break;
            }
            
            // Extract context/state and player_input
            if (!request.contains("player_input")) {
                json error_response = {
                    {"response", "Missing 'player_input' field"},
                    {"npc_id", ""},
                    {"success", false}
                };
                std::cout << error_response.dump() << std::endl;
                std::cout.flush();
                continue;
            }
            
            std::string player_input = request["player_input"];
            std::string response;
            std::string npc_id = "NPC";
            
            if (request.contains("game_state")) {
                // Advanced mode: Update state directly
                json game_state = request["game_state"];
                npc_id = game_state.value("npc_id", "NPC");
                
                engine.UpdateState(game_state);
                response = engine.Generate(player_input);
                
            } else if (request.contains("context")) {
                // Legacy/Simplified mode
                json context = request["context"];
                npc_id = context.value("npc_id", "NPC");
                std::string persona = context.value("persona", "You are a helpful NPC.");
                std::string scenario = context.value("scenario", "");
                
                response = engine.GenerateFromContext(
                    persona,
                    npc_id,
                    scenario,
                    player_input
                );
            } else {
                 // Fallback or error
                 response = "Error: Missing context or game_state";
            }
            
            // Output JSON response
            json json_response = {
                {"response", response},
                {"npc_id", npc_id},
                {"success", true}
            };
            std::cout << json_response.dump() << std::endl;
            std::cout.flush();
            
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            json error_response = {
                {"response", std::string("JSON parse error: ") + e.what()},
                {"npc_id", ""},
                {"success", false}
            };
            std::cout << error_response.dump() << std::endl;
            std::cout.flush();
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            json error_response = {
                {"response", std::string("Error: ") + e.what()},
                {"npc_id", ""},
                {"success", false}
            };
            std::cout << error_response.dump() << std::endl;
            std::cout.flush();
        }
    }
    
    return 0;
}
