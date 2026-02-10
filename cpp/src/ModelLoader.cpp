// ModelLoader.cpp - Implementation of ONNX model loading and inference

#include "ModelLoader.h"
#include "KVCacheManager.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <cmath>
#include <utility>

namespace NPCInference {

// PIMPL struct definition for KV Cache
struct ModelLoader::KVCache {
    std::vector<Ort::Value> values;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
};

ModelLoader::ModelLoader() 
    : is_loaded_(false) {
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NPCInference");
    kv_cache_ = std::make_unique<KVCache>();
    cache_manager_ = std::make_shared<KVCacheManager>(512, 100); // 512MB, 100 conversations
}

ModelLoader::~ModelLoader() = default;

bool ModelLoader::LoadModel(const std::string& model_path, bool use_cuda, int num_threads) {
    try {
        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(num_threads > 0 ? num_threads : 4);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Enable CUDA if requested
        if (use_cuda) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options_->AppendExecutionProvider_CUDA(cuda_options);
                std::cerr << "CUDA provider enabled" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "CUDA not available, falling back to CPU: " << e.what() << std::endl;
            }
        }
        
        // Load the model
#ifdef _WIN32
        // Windows uses wide strings for file paths
        std::wstring wide_path(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), *session_options_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
#endif
        
        // Introspect model IO
        Ort::AllocatorWithDefaultOptions allocator;
        
        kv_cache_->input_names.clear();
        kv_cache_->output_names.clear();
        kv_cache_->input_name_ptrs.clear();
        kv_cache_->output_name_ptrs.clear();

        size_t num_input_nodes = session_->GetInputCount();
        for(size_t i = 0; i < num_input_nodes; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            kv_cache_->input_names.push_back(name.get());
        }
        
        size_t num_output_nodes = session_->GetOutputCount();
        for(size_t i = 0; i < num_output_nodes; i++) {
             auto name = session_->GetOutputNameAllocated(i, allocator);
             kv_cache_->output_names.push_back(name.get());
        }
        
        // Pre-compute pointers for Run()
        for(const auto& name : kv_cache_->input_names) kv_cache_->input_name_ptrs.push_back(name.c_str());
        for(const auto& name : kv_cache_->output_names) kv_cache_->output_name_ptrs.push_back(name.c_str());
        
        is_loaded_ = true;
        std::cerr << "Model loaded successfully. Inputs: " << num_input_nodes << ", Outputs: " << num_output_nodes << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        is_loaded_ = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        is_loaded_ = false;
        return false;
    }
}

std::vector<int64_t> ModelLoader::Generate(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    int max_new_tokens,
    const std::string& conversation_id,
    std::function<void(int64_t)> on_token_callback
) {
    if (!is_loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    // Try to load cached KV from previous turns
    bool using_cache = false;
    size_t cached_seq_len = 0;
    
    if (!conversation_id.empty() && cache_manager_) {
        auto* cached = cache_manager_->Get(conversation_id);
        if (cached) {
            // Restore cached KV tensors
            kv_cache_->values = std::move(cached->kv_tensors);
            cached_seq_len = cached->sequence_length;
            using_cache = true;
            std::cerr << "KVCache: Restored cache for '" << conversation_id 
                      << "', seq_len=" << cached_seq_len << std::endl;
        }
    }
    
    // If no cache, clear any stale values
    if (!using_cache) {
        kv_cache_->values.clear();
    }
    
    // Prepare memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, 
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<int64_t> current_input_ids = input_ids;
    std::vector<int64_t> generated_ids = input_ids;
    
    // Random number generator for sampling
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < max_new_tokens; ++i) {
        // Prepare inputs
        std::vector<const char*> run_input_names;
        std::vector<Ort::Value> run_input_values; 
        
        // 1. input_ids
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(current_input_ids.size())};
        
        // If we have KV cache, we only pass the LAST token, so shape is [1, 1]
        if (!kv_cache_->values.empty()) {
             input_shape = {1, 1};
        }
        
        run_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            const_cast<int64_t*>(current_input_ids.data()),
            input_shape[1],
            input_shape.data(),
            input_shape.size()
        ));
        run_input_names.push_back("input_ids");
        
        // 2. Add attention_mask if model expects it
        bool needs_mask = false;
        for(const auto& name : kv_cache_->input_names) {
            if (name == "attention_mask") needs_mask = true;
        }
        
        if (needs_mask) {
            std::vector<int64_t> total_mask(generated_ids.size(), 1); 
            std::vector<int64_t> mask_shape = {1, static_cast<int64_t>(total_mask.size())};
            
            run_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info,
                total_mask.data(),
                total_mask.size(),
                mask_shape.data(),
                mask_shape.size()
            ));
            run_input_names.push_back("attention_mask");
        }

        // 3. Add KV cache tensors
        if (!kv_cache_->values.empty()) {
            int val_idx = 0;
            for(const auto& name : kv_cache_->input_names) {
                if (name.find("past_") != std::string::npos && val_idx < kv_cache_->values.size()) {
                    run_input_names.push_back(name.c_str());
                    run_input_values.push_back(std::move(kv_cache_->values[val_idx]));
                    val_idx++;
                }
            }
        }

        // Prepare outputs
        std::vector<const char*> run_output_names = kv_cache_->output_name_ptrs;
        
        try {
            auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr},
                run_input_names.data(),
                run_input_values.data(),
                run_input_values.size(),
                run_output_names.data(),
                run_output_names.size()
            );
            
            // 1. Process Logits (usually first output)
            float* logits = ort_outputs[0].GetTensorMutableData<float>();
            auto shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            
            int64_t batch_size = shape[0];
            int64_t seq_len = shape[1]; 
            int64_t vocab_size = shape[2];
            
            float* last_logits = logits + (batch_size * seq_len - 1) * vocab_size;
            int64_t next_token = SampleToken(last_logits, vocab_size);
            
            // EOS Check
            if (next_token == 32000 || next_token == 32007) break;
            
            generated_ids.push_back(next_token);
            
            // Streaming callback
            if (on_token_callback) {
                on_token_callback(next_token);
            }
            
            // 2. Update KV Cache
            kv_cache_->values.clear();
            for(size_t k = 1; k < ort_outputs.size(); ++k) {
                kv_cache_->values.push_back(std::move(ort_outputs[k]));
            }
            
            // 3. Prepare for next iteration
            if (!kv_cache_->values.empty()) {
                current_input_ids.clear();
                current_input_ids.push_back(next_token);
            } else {
                current_input_ids = generated_ids;
            }
            
        } catch (const Ort::Exception& e) {
             std::cerr << "Inference Error: " << e.what() << std::endl;
             break;
        }
    }

    // Save KV cache for next turn
    if (!conversation_id.empty() && cache_manager_ && !kv_cache_->values.empty()) {
        cache_manager_->Put(conversation_id, 
                           std::move(kv_cache_->values),
                           generated_ids.size());
        std::cerr << "KVCache: Saved cache for '" << conversation_id 
                  << "', seq_len=" << generated_ids.size() << std::endl;
    }

    return generated_ids;
}

void ModelLoader::ClearCache(const std::string& conversation_id) {
    if (!cache_manager_) return;
    
    if (conversation_id.empty()) {
        cache_manager_->Clear();
        std::cerr << "KVCache: Cleared all caches" << std::endl;
    } else {
        cache_manager_->Remove(conversation_id);
        std::cerr << "KVCache: Cleared cache for '" << conversation_id << "'" << std::endl;
    }
}

void ModelLoader::PrintCacheStats() const {
    if (!cache_manager_) return;
    
    auto stats = cache_manager_->GetStats();
    std::cerr << "\n=== KV-Cache Statistics ===" << std::endl;
    std::cerr << "Total Entries: " << stats.total_entries << std::endl;
    std::cerr << "Memory Usage: " << (stats.total_memory_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cerr << "Cache Hits: " << stats.hits << std::endl;
    std::cerr << "Cache Misses: " << stats.misses << std::endl;
    std::cerr << "Hit Rate: " << (stats.hit_rate() * 100.0f) << "%" << std::endl;
    std::cerr << "Evictions: " << stats.evictions << std::endl;
    std::cerr << "==========================\n" << std::endl;
}

int64_t ModelLoader::SampleToken(float* logits, int64_t vocab_size) {
    // 1. Temperature
    if (temperature_ > 0.0f) {
        for (int64_t i = 0; i < vocab_size; ++i) {
            logits[i] /= temperature_;
        }
    }
    
    // Softmax
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) probs[i] /= sum_exp;
    
    // 2. Top-P (Nucleus) Sampling
    if (top_p_ < 1.0f && top_p_ > 0.0f) {
        // Pair prob with index
        std::vector<std::pair<float, int64_t>> prob_indices(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            prob_indices[i] = {probs[i], i};
        }
        
        // Sort descending
        std::sort(prob_indices.begin(), prob_indices.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
                 
        float cum_prob = 0.0f;
        int64_t cutoff_index = vocab_size - 1;
        
        for (int64_t i = 0; i < vocab_size; ++i) {
            cum_prob += prob_indices[i].first;
            if (cum_prob > top_p_) {
                cutoff_index = i;
                break;
            }
        }
        
        // Filter and renormalize
        std::vector<float> filtered_probs;
        std::vector<int64_t> filtered_indices;
        float new_sum = 0.0f;
        
        for (int64_t i = 0; i <= cutoff_index; ++i) {
            filtered_probs.push_back(prob_indices[i].first);
            filtered_indices.push_back(prob_indices[i].second);
            new_sum += prob_indices[i].first;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, new_sum);
        float r = dis(gen);
        
        float acc = 0.0f;
        for (size_t i = 0; i < filtered_probs.size(); ++i) {
            acc += filtered_probs[i];
            if (acc >= r) return filtered_indices[i];
        }
        return filtered_indices.back();
    }
    
    // Fallback: Argmax
    return std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
}

} // namespace NPCInference
