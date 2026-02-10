#pragma once

#include <string>
#include <vector>
#include <memory>
#include "Tokenizer.h"

namespace NPCInference {

    class ModelLoader; // Forward decl if needed, but we might implement a separate loader for embeddings.
    
    // Actually, ModelLoader in NPCInference is specifically for CausalLM (GenAI).
    // EmbeddingModel is simpler (Encoder-only).
    
    class EmbeddingModel {
    public:
        EmbeddingModel();
        ~EmbeddingModel();

        // Load ONNX model and tokenizer
        bool Load(const std::string& model_path, const std::string& tokenizer_path);

        // Generate embedding for text
        std::vector<float> Embed(const std::string& text);

        bool IsLoaded() const { return loaded_; }

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        
        std::unique_ptr<Tokenizer> tokenizer_;
        bool loaded_ = false;
    };

} // namespace NPCInference
