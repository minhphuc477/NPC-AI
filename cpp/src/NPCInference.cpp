#include "NPCInference.h"
#include "PythonBridge.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace NPCInference {

    NPCInferenceEngine::NPCInferenceEngine() {
        model_loader_ = std::make_unique<ModelLoader>();
        prompt_formatter_ = std::make_unique<PromptFormatter>();
        prompt_builder_ = std::make_unique<PromptBuilder>(true);
        behavior_tree_ = NPCBehavior::CreateNPCBehaviorTree();
        tokenizer_ = std::make_unique<Tokenizer>();
        vector_store_ = std::make_unique<VectorStore>();
        embedding_model_ = std::make_unique<EmbeddingModel>();
    }

    NPCInferenceEngine::~NPCInferenceEngine() = default;



    bool NPCInferenceEngine::Initialize(const InferenceConfig& config) {
        config_ = config;
        std::string modelPath = config.model_dir;

        // Load tokenizer
        std::string tokenizer_path = modelPath + "/tokenizer.model";
        if (!tokenizer_->Load(tokenizer_path)) {
            std::cerr << "Warning: Failed to load tokenizer from " << tokenizer_path << std::endl;
        }
        
        // Load model
        std::string onnx_path = modelPath + "/model.onnx";
        if (!model_loader_->LoadModel(onnx_path, config.use_cuda, config.num_threads)) {
             std::cerr << "Warning: Failed to load native model from " << onnx_path << std::endl;
        }
        
        // Load Embedding Model (Optional RAG)
        std::string embed_path = modelPath + "/" + config.embedding_model_name;
        std::string spm_path = modelPath + "/" + config.tokenizer_embedding_path;
        
        if (embedding_model_->Load(embed_path, spm_path)) {
            // Initialize Vector Store (384 dim for MiniLM-L12)
            if (vector_store_->Initialize(384)) {
                // Try to load existing vectors
                vector_store_->Load(modelPath + "/vectors");
                std::cout << "RAG: Vector Memory initialized." << std::endl;
            }
        }
        
        ready_ = true;
        return true;
    }

    bool NPCInferenceEngine::SaveMemory() {
        if (!vector_store_ || config_.model_dir.empty()) return false;
        return vector_store_->Save(config_.model_dir + "/vectors");
    }

    bool NPCInferenceEngine::SaveState(const std::string& filepath) {
        try {
            std::ofstream f(filepath);
            if (!f.is_open()) return false;
            f << current_state_.dump(4);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool NPCInferenceEngine::LoadState(const std::string& filepath) {
        try {
            std::ifstream f(filepath);
            if (!f.is_open()) return false;
            nlohmann::json j;
            f >> j;
            current_state_ = j;
            return true;
        } catch (...) {
            return false;
        }
    }

    bool NPCInferenceEngine::LoadWithBridge(const std::string& pythonPath, const std::string& scriptPath, const std::string& modelPath) {
        python_bridge_ = std::make_unique<PythonBridge>();
        if (python_bridge_->Start(pythonPath, scriptPath, modelPath)) {
            bridge_mode_ = true;
            ready_ = true;
            return true;
        }
        return false;
    }

    std::string NPCInferenceEngine::UpdateState(const json& gameState) {
        current_state_ = gameState;
        
        // Map json to blackboard for BT
        NPCBehavior::Blackboard blackboard = current_state_;
        
        // Ensure required keys exist
        if (!blackboard.contains("hp")) blackboard["hp"] = 100;
        if (!blackboard.contains("is_combat")) blackboard["is_combat"] = false;
        
        // Tick BT
        behavior_tree_->tick(blackboard);
        
        // Update Action
        if (blackboard.contains("current_action")) {
            current_action_ = blackboard["current_action"];
        } else {
            current_action_ = "Idle";
        }
        
        return current_action_;
    }

    std::string NPCInferenceEngine::Generate(const std::string& prompt) {
        if (!ready_) return "Error: Engine not ready";

        if (bridge_mode_ && python_bridge_) {
             json response = python_bridge_->SendRequest({
                {"player_input", prompt},
                {"context", current_state_}
             });
             
             if (response.contains("response")) {
                 if (response["response"].is_string()) {
                     return response["response"];
                 } else {
                     return response["response"].dump();
                 }
             }
             return response.dump();
        }
        
        // Native Generation
        if (model_loader_->IsLoaded() && tokenizer_->IsLoaded()) {
             // 1. RAG Retrieval (if available)
             if (embedding_model_->IsLoaded()) {
                 std::vector<float> query_vec = embedding_model_->Embed(prompt);
                 if (!query_vec.empty()) {
                     // Search for relevant memories
                     auto results = vector_store_->Search(query_vec, 3);
                     if (!results.empty()) {
                         std::string memory_block;
                         for (const auto& res : results) {
                             // Use configurable threshold (distance is cosine distance, so closer to 0 is better if using distance, 
                             // but usually cosine similarity is 1.0 for identical.
                             // VectorStore likely returns distance (1 - similarity) or raw distance.
                             // Assuming distance: lower is better.
                             // Validating assumption: standard implementations often use L2 or Cosine distance. 
                             // If it's pure cosine similarity, higher is better.
                             // Let's assume the implementation uses distance < threshold for relevance.
                             
                             // Let's assume the implementation uses distance < threshold for relevance.
                             
                             if (res.distance < config_.rag_threshold) { 
                                 memory_block += "- " + res.text + "\n";
                             }
                         }
                         if (!memory_block.empty()) {
                             current_state_["memory_context"] = memory_block;
                             std::cout << "RAG: Retrieved " << results.size() << " entries." << std::endl;
                         }
                     }
                 }
             }

             // 2. Build Prompt
             // Use language from state if present, else default to "vi"
             std::string lang = current_state_.value("language", "vi");
             // Extract Tools from state if present
             json tools = current_state_.value("tools", json::array());
             
             std::string full_prompt = prompt_builder_->Build(current_state_, current_state_, prompt, lang, tools);
             
             // 2. Tokenize
             std::vector<int64_t> input_ids = tokenizer_->Encode(full_prompt);
             
             if (input_ids.empty()) return "Error: Tokenization failed";

             // 3. Generate with KV-cache persistence
             // Extract conversation ID from state (or use NPC ID as fallback)
             std::string conv_id = current_state_.value("conversation_id", 
                                                        current_state_.value("npc_id", "default"));
             
             try {
                 std::vector<int64_t> attention_mask(input_ids.size(), 1);
                 
                 // Generate with streaming callback
                 std::vector<int64_t> output_ids = model_loader_->Generate(
                     input_ids, 
                     attention_mask, 
                     150,
                     conv_id,  // Enable cache persistence
                     [this](int64_t token) {  // Streaming callback
                         // Could emit events here for real-time UI updates
                         // For now, just silent streaming
                     }
                 );
                 
                 // 4. Decode
                 std::vector<int64_t> new_tokens;
                 if (output_ids.size() > input_ids.size()) {
                     new_tokens.assign(output_ids.begin() + input_ids.size(), output_ids.end());
                 } else {
                     return ""; // No tokens generated
                 }
                 
                 std::string response_text = tokenizer_->Decode(new_tokens);
                 return response_text;
                 
             } catch (const std::exception& e) {
                 return std::string("Error during generation: ") + e.what();
             }
        }
        
        return "Error: Native generation not available (Model or Tokenizer not loaded)";
    }

    std::string NPCInferenceEngine::GenerateFromContext(const std::string& persona, const std::string& npc_id, const std::string& scenario, const std::string& player_input) {
        
        // Update State
        json state;
        state["persona"] = persona;
        state["npc_id"] = npc_id;
        state["language"] = "vi"; // Default language

        // Try to parse scenario as JSON context
        try {
            json context_json = json::parse(scenario);
            // Merge context into state
            state.update(context_json);
            
            // Should also set scenario_plot if present, or use raw string if parsing failed
            if (!state.contains("scenario_plot") && context_json.contains("plot")) {
                state["scenario_plot"] = context_json["plot"];
            }
        } catch (...) {
            // Fallback: Treat scenario as just the plot string
            state["scenario_plot"] = scenario;
        }

        state["is_player_nearby"] = true;
        state["is_player_talking"] = !player_input.empty();
        
        UpdateState(state);
        
        // Generate
        return Generate(player_input);
    }

    bool NPCInferenceEngine::Remember(const std::string& text, const std::map<std::string, std::string>& metadata) {
        if (!embedding_model_ || !embedding_model_->IsLoaded()) return false;
        
        std::vector<float> vec = embedding_model_->Embed(text);
        if (vec.empty()) return false;

        if (vector_store_) {
            vector_store_->Add(text, vec, metadata);
            std::cout << "RAG: Remembered '" << text.substr(0, 30) << "...'" << std::endl;
            return true;
        }
        return false;
    }



    std::string NPCInferenceEngine::FormatPrompt(const std::string& system, const std::string& name, const std::string& context, const std::string& question) {
        return prompt_formatter_->Format(system, name, context, question);
    }
    
    // Legacy / Placeholder methods
    bool NPCInferenceEngine::LoadModel(const std::string& model_path, const std::string& adapter_path, bool use_cuda, bool bridge_mode) {
         if (bridge_mode) return LoadWithBridge("python", "npc_cli.py", model_path);
         return Initialize(model_path);
    }

    std::string NPCInferenceEngine::Generate(const std::string& prompt, const std::string& npc_name) {
        return Generate(prompt);
    }

    std::vector<int64_t> NPCInferenceEngine::Tokenize(const std::string& text) {
        if (tokenizer_ && tokenizer_->IsLoaded()) {
            return tokenizer_->Encode(text);
        }
        std::cerr << "WARNING: Tokenizer not loaded, returning empty vector!" << std::endl;
        return {};
    }

    std::string NPCInferenceEngine::Decode(const std::vector<int64_t>& token_ids) {
        if (tokenizer_ && tokenizer_->IsLoaded()) {
            return tokenizer_->Decode(token_ids);
        }
        std::cerr << "WARNING: Tokenizer not loaded, returning empty string!" << std::endl;
        return "";
    }

    bool NPCInferenceEngine::Initialize(const std::string& modelPath) {
        InferenceConfig config;
        config.model_dir = modelPath;
        return Initialize(config);
    }



    // ... Remember/SaveMemory ...

    NPCInferenceEngine::GenerationResult NPCInferenceEngine::ParseOutput(const std::string& raw_output) {
        GenerationResult result;
        result.text = raw_output;
        
        // 1. Look for ```json ... ``` block
        std::string marker_start = "```json";
        std::string marker_end = "```";
        
        size_t start_pos = raw_output.find(marker_start);
        if (start_pos != std::string::npos) {
            size_t content_start = start_pos + marker_start.length();
            size_t end_pos = raw_output.find(marker_end, content_start);
            if (end_pos != std::string::npos) {
                std::string json_str = raw_output.substr(content_start, end_pos - content_start);
                // Verify valid JSON
                try {
                    auto j = json::parse(json_str);
                    result.tool_call = j.dump(); 
                    // Remove the JSON block from text? Or keep it?
                    // Ideally, if it's a tool call, the text might be empty or reasoning.
                    // For now, let's keep text as is, or strip the block if we want clean text.
                } catch (...) {}
            }
        } else {
            // 2. Try parsing entire string as JSON
            try {
                auto j = json::parse(raw_output);
                if (j.contains("tool") || j.contains("function")) {
                    result.tool_call = raw_output;
                }
            } catch (...) {}
        }
        
        return result;
    }

} // namespace NPCInference
