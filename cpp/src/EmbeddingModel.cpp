#ifdef _WIN32
#include <windows.h>
#undef GetCurrentTime
#endif

#include "EmbeddingModel.h"
#include "OrtEnvironmentManager.h"
#include "NPCLogger.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

namespace NPCInference {

    struct EmbeddingModel::Impl {
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::SessionOptions> session_options;
        Ort::AllocatorWithDefaultOptions allocator;
    };

    EmbeddingModel::EmbeddingModel() : impl_(std::make_unique<Impl>()) {
        tokenizer_ = std::make_unique<Tokenizer>();
    }

    EmbeddingModel::~EmbeddingModel() = default;

    bool EmbeddingModel::Load(const std::string& model_path, const std::string& tokenizer_path) {
        // Check for Mock Mode
        const char* mock_env = std::getenv("NPC_MOCK_MODE");
        if (mock_env && std::string(mock_env) == "1") {
            NPCLogger::Info("EmbeddingModel: Running in MOCK MODE - Weights skipped.");
            loaded_ = true;
            return true;
        }
        
        try {
            // Load Tokenizer
            if (!tokenizer_->Load(tokenizer_path)) {
                std::cerr << "Failed to load tokenizer for embeddings." << std::endl;
                return false;
            }

            // Load ONNX Model
            impl_->session_options = std::make_unique<Ort::SessionOptions>();
            impl_->session_options->SetIntraOpNumThreads(1);
            impl_->session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            Ort::Env& env = OrtEnvironmentManager::Instance().GetEnv();
#ifdef _WIN32
            std::wstring wide_path(model_path.begin(), model_path.end());
            impl_->session = std::make_unique<Ort::Session>(env, wide_path.c_str(), *impl_->session_options);
#else
            impl_->session = std::make_unique<Ort::Session>(env, model_path.c_str(), *impl_->session_options);
#endif
            loaded_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "EmbeddingModel Load Error: " << e.what() << std::endl;
            loaded_ = false;
            return false;
        }
    }

    std::vector<float> EmbeddingModel::Embed(const std::string& text) {
        if (!loaded_) return {};

        // Mock mode: return deterministic embeddings based on text hash
        const char* mock_env = std::getenv("NPC_MOCK_MODE");
        if (mock_env && std::string(mock_env) == "1") {
            const size_t mock_dim = 768; // Standard BERT dimension
            std::vector<float> mock_embedding(mock_dim, 0.0f);
            
            // Simple hash-based mock embedding
            std::hash<std::string> hasher;
            size_t hash = hasher(text);
            
            for (size_t i = 0; i < mock_dim; ++i) {
                mock_embedding[i] = static_cast<float>((hash + i) % 1000) / 1000.0f - 0.5f;
            }
            
            // Normalize
            float norm = 0.0f;
            for (float val : mock_embedding) norm += val * val;
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (float& val : mock_embedding) val /= norm;
            }
            
            return mock_embedding;
        }

        // 1. Tokenize
        std::vector<int64_t> input_ids = tokenizer_->Encode(text);
        if (input_ids.empty()) return {};

        // 2. Prepare Inputs
        // input_ids shape: [1, seq_len]
        // attention_mask shape: [1, seq_len]
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
        std::vector<int64_t> attention_mask(input_ids.size(), 1);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));
        
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()));

        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"embeddings"};

        try {
            auto output_tensors = impl_->session->Run(
                Ort::RunOptions{nullptr},
                input_names,
                input_tensors.data(),
                2,
                output_names,
                1
            );

            // Get output
            // Shape: [1, hidden_size]
            float* floatarr = output_tensors[0].GetTensorMutableData<float>();
            auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t dim = shape[1]; // Usually 384 or 768 or 512

            std::vector<float> embedding(floatarr, floatarr + dim);
            return embedding;

        } catch (const std::exception& e) {
            std::cerr << "Embedding Inference Error: " << e.what() << std::endl;
            return {};
        }
    }

} // namespace NPCInference
