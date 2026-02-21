// ModelLoader.cpp - Implementation of ONNX model loading and inference

// NOMINMAX defined in CMake
#ifdef _WIN32
#include <windows.h>
#undef GetCurrentTime
#endif

#include "ModelLoader.h"
#include "KVCacheManager.h"
#include "OrtEnvironmentManager.h"
#include "NPCLogger.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <random>
#include <cmath>
#include <utility>

namespace NPCInference {

// PIMPL struct definition for KV Cache
struct ModelLoader::KVCache {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    std::vector<Ort::Value> values;
};

ModelLoader::ModelLoader() 
    : is_loaded_(false) {
    // Shared ONNX Runtime environment is managed by OrtEnvironmentManager
    kv_cache_ = std::make_unique<KVCache>();
    cache_manager_ = std::make_shared<KVCacheManager>(512, 100); // 512MB, 100 conversations
    
    std::random_device rd;
    gen_.seed(rd());
}

ModelLoader::~ModelLoader() = default;

bool ModelLoader::LoadModel(const std::string& model_path, bool use_cuda, int num_threads) {
    // Check for MOCK MODE
    const char* mock_env = std::getenv("NPC_MOCK_MODE");
    if (mock_env && std::string(mock_env) == "1") {
        NPCLogger::Warn("!! RUNNING IN MOCK MODE !! - Logic verified, Weights skipped.");
        is_loaded_ = true;
        // session_ remains nullptr
        
        // Setup dummy KV cache config
        kv_cache_->input_names = {"input_ids", "attention_mask", "past_key_values"};
        kv_cache_->output_names = {"logits", "present_key_values"};
        is_mock_ = true;
        return true;
    }

    try {
        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        
        // Phase 8 Resource Contention Fix: Clamp threads
        // Reserve at least half the CPU cores for Unreal Engine (Physics, Render, Game threads)
        int hw_cores = std::thread::hardware_concurrency();
        int safe_threads = std::max(1, hw_cores > 4 ? hw_cores / 2 : hw_cores - 1);
        int final_threads = (num_threads > 0) ? std::min(num_threads, safe_threads) : std::min(4, safe_threads);
        
        session_options_->SetIntraOpNumThreads(final_threads);
        session_options_->SetInterOpNumThreads(1); // Keep inter-op logic strictly linear to avoid cache thrash
        
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Enable CUDA if requested
        if (use_cuda) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                cuda_options.gpu_mem_limit = static_cast<size_t>(-1); // Use all available
                cuda_options.arena_extend_strategy = 0; 
                cuda_options.do_copy_in_default_stream = 1;
                cuda_options.has_user_compute_stream = 0;
                
                session_options_->AppendExecutionProvider_CUDA(cuda_options);
                
                // Set optimization level to full (to allow attention fusion)
                session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                
                std::cerr << "CUDA provider enabled with advanced optimizations" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "CUDA not available, falling back to CPU: " << e.what() << std::endl;
            }
        }
        
        // Load the model
#ifdef _WIN32
        // Windows uses wide strings for file paths
        std::wstring wide_path(model_path.begin(), model_path.end());
        Ort::Env& env = OrtEnvironmentManager::Instance().GetEnv();
        session_ = std::make_unique<Ort::Session>(env, wide_path.c_str(), *session_options_);
#else
        Ort::Env& env = OrtEnvironmentManager::Instance().GetEnv();
        session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), *session_options_);
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
    std::function<void(int64_t)> on_token_callback,
    std::function<void(float*, int64_t)> logit_processor
) {
    if (is_mock_) {
         std::cerr << "DEBUG: Mock Generate sleeping..." << std::endl;
         std::this_thread::sleep_for(std::chrono::milliseconds(50 + (max_new_tokens * 2))); 
         std::vector<int64_t> mock_out;
         for(int i=0; i<max_new_tokens; i++) {
             mock_out.push_back(100+i);
             if(on_token_callback) on_token_callback(100+i);
         }
         std::cerr << "DEBUG: Mock Generate done. Tokens: " << mock_out.size() << std::endl;
         return mock_out;
    }

    if (!is_loaded_) {
        std::cerr << "Generate: Model not loaded!" << std::endl;
        return {};
    }
    std::cerr << "Generate: processing " << input_ids.size() << " input tokens." << std::endl;

    // Try to load cached KV from previous turns
    bool using_cache = false;
    size_t cached_seq_len = 0;
    
    
    // Local container for KV tensors to ensure thread-safety
    std::vector<Ort::Value> current_kv_tensors;
    
    if (!conversation_id.empty() && cache_manager_) {
        auto cached = cache_manager_->Get(conversation_id);
        if (cached) {
            // Restore cached KV tensors
            current_kv_tensors = std::move(cached->kv_tensors);
            cached_seq_len = cached->sequence_length;
            using_cache = true;
            std::cerr << "KVCache: Restored cache for '" << conversation_id 
                      << "', seq_len=" << cached_seq_len << std::endl;
        } else {
             // Try System Prompt Cache (Optimization)
             auto system_cache = cache_manager_->GetSystemKV();
             if (system_cache) {
                 current_kv_tensors = std::move(system_cache->kv_tensors);
                 cached_seq_len = system_cache->sequence_length;
                 using_cache = true;
             }
        }
    }
    
    // Prepare memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, 
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<int64_t> current_input_ids = input_ids;
    std::vector<int64_t> generated_ids = input_ids;
    
    // MOCK MODE: Return dummy tokens if session is null
    if (!session_) {
        // Simulate processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // ~50 tok/s
        
        for (int i = 0; i < max_new_tokens; ++i) {
             generated_ids.push_back(100 + i); // Dummy token ID
             if (on_token_callback) on_token_callback(100 + i);
        }
        return generated_ids;
    }

    // Reset flag at start
    {
        std::lock_guard<std::mutex> lock(cancel_mutex_);
        cancel_flags_[conversation_id] = false;
    }

    for (int i = 0; i < max_new_tokens; ++i) {
        
        // Immediate termination point
        bool is_cancelled = false;
        {
            std::lock_guard<std::mutex> lock(cancel_mutex_);
            is_cancelled = cancel_flags_[conversation_id];
        }
        
        if (is_cancelled) {
            std::cerr << "ModelLoader: Generation cancelled internally for conv: " << conversation_id << std::endl;
            break;
        }

        // Prepare inputs
        std::vector<const char*> run_input_names;
        std::vector<Ort::Value> run_input_values; 
        
        std::lock_guard<std::mutex> kv_lock(kv_mutex_);
        
        // 1. input_ids
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(current_input_ids.size())};
        
        // If we have KV cache, we only pass the LAST token, so shape is [1, 1]
        if (!kv_cache_->values.empty()) {
             input_shape = {1, 1};
        }
        
        // Use mutable copy to avoid const_cast
        std::vector<int64_t> mutable_ids = current_input_ids;
        run_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            mutable_ids.data(),
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
        if (!current_kv_tensors.empty()) {
            int val_idx = 0;
            for(const auto& name : kv_cache_->input_names) {
                if (name.find("past_") != std::string::npos && val_idx < current_kv_tensors.size()) {
                    run_input_names.push_back(name.c_str());
                    run_input_values.push_back(std::move(current_kv_tensors[val_idx]));
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
            
            // Apply Processor (Phase 12)
            if (logit_processor) {
                logit_processor(last_logits, vocab_size);
            }

            int64_t next_token = SampleToken(last_logits, vocab_size);
            
            // EOS Check
            if (next_token == 32000 || next_token == 32007) break;
            
            generated_ids.push_back(next_token);
            
            // Streaming callback
            if (on_token_callback) {
                on_token_callback(next_token);
            }
            
            // 2. Update KV Cache
            current_kv_tensors.clear();
            for(size_t k = 1; k < ort_outputs.size(); ++k) {
                current_kv_tensors.push_back(std::move(ort_outputs[k]));
            }
            
            // 3. Prepare for next iteration
            if (!current_kv_tensors.empty()) {
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
    if (!conversation_id.empty() && cache_manager_ && !current_kv_tensors.empty()) {
        cache_manager_->Put(conversation_id, 
                           std::move(current_kv_tensors),
                           generated_ids.size());
    }

    return generated_ids;
}

std::vector<int64_t> ModelLoader::VerifyDraft(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& draft_ids,
    const std::string& conversation_id
) {
    if (!is_loaded_ || draft_ids.empty()) return {};

    // MOCK MODE
    if (!session_) {
        // In mock mode, we verify everything (or random)
        // Let's accept all draft tokens to simulate perfect speculation speedup?
        return draft_ids;
    }

    bool using_cache = false;
    if (!conversation_id.empty() && cache_manager_) {
        auto cached = cache_manager_->Get(conversation_id);
        if (cached) {
            kv_cache_->values = std::move(cached->kv_tensors);
            using_cache = true;
        }
    }
    
    if (!using_cache) kv_cache_->values.clear();

    std::lock_guard<std::mutex> kv_lock(kv_mutex_);

    // 2. Prepare Inputs for Verification
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, 
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<int64_t> tokens_to_process = draft_ids;
    if (!using_cache && !input_ids.empty()) {
        tokens_to_process = input_ids;
        tokens_to_process.insert(tokens_to_process.end(), draft_ids.begin(), draft_ids.end());
    }

    // Run inference
    std::vector<int64_t> accepted_tokens;
    std::vector<const char*> run_input_names;
    std::vector<Ort::Value> run_input_values;

    // Create Input Tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens_to_process.size())};
    std::vector<int64_t> mutable_tokens = tokens_to_process;
    run_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, mutable_tokens.data(),
        input_shape[1], input_shape.data(), input_shape.size()
    ));
    run_input_names.push_back("input_ids");

    // Mask
    bool needs_mask = false;
    for(const auto& name : kv_cache_->input_names) if(name == "attention_mask") needs_mask = true;
    if (needs_mask) {
        std::vector<int64_t> mask(tokens_to_process.size(), 1); 
        std::vector<int64_t> mask_shape = {1, static_cast<int64_t>(mask.size())};
        run_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, mask.data(), mask.size(), mask_shape.data(), mask_shape.size()
        ));
        run_input_names.push_back("attention_mask");
    }

    // KV Cache Inputs
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

    try {
        auto ort_outputs = session_->Run(
            Ort::RunOptions{nullptr},
            run_input_names.data(), run_input_values.data(), run_input_values.size(),
            kv_cache_->output_name_ptrs.data(), kv_cache_->output_name_ptrs.size()
        );

        // 3. Verification Loop
        float* logits = ort_outputs[0].GetTensorMutableData<float>();
        auto shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t seq_len = shape[1]; 
        int64_t vocab_size = shape[2];
        
        // --- PHASE 7: SPECULATIVE MISMATCH GUARD ---
        for (int64_t d_id : draft_ids) {
            if (d_id < 0 || d_id >= vocab_size) {
                std::cerr << "Speculative Mismatch Guard: Draft model token (" << d_id << ") exceeds Target vocab size (" << vocab_size << "). Different architectures detected! Aborting draft." << std::endl;
                return {};
            }
        }
        
        int64_t check_start_index = 0;
        if (!using_cache && !input_ids.empty()) {
            check_start_index = input_ids.size() - 1;
        }

        bool all_matched = true;
        int mismatch_idx = -1;
        int64_t correction_token = -1;

        for (size_t i = 0; i < draft_ids.size(); ++i) {
            size_t logit_idx = check_start_index + i;
            
            if (logit_idx >= seq_len) {
                std::cerr << "Error: logit index " << logit_idx << " out of bounds" << std::endl;
                break;
            }

            float* token_logits = logits + (logit_idx * vocab_size);
            int64_t main_token = std::distance(token_logits, std::max_element(token_logits, token_logits + vocab_size));

            if (i + 1 < draft_ids.size()) {
                if (main_token == draft_ids[i+1]) {
                    accepted_tokens.push_back(main_token);
                } else {
                    correction_token = main_token;
                    mismatch_idx = i;
                    all_matched = false;
                    break;
                }
            } else {
                correction_token = main_token;
                accepted_tokens.push_back(correction_token);
            }
        }

        // 4. Correct Cache State
        if (!all_matched || mismatch_idx != -1) {
             if (!conversation_id.empty() && cache_manager_) {
                  auto clean = cache_manager_->Get(conversation_id);
                  if (clean) {
                       kv_cache_->values = std::move(clean->kv_tensors);
                  }
             }
        } else {
             if (draft_ids.size() < (size_t)seq_len) {
                  size_t last_idx = draft_ids.size() - 1;
                  if (last_idx < (size_t)seq_len) {
                      float* last_logits = logits + (last_idx * vocab_size);
                      correction_token = SampleToken(last_logits, vocab_size);
                      accepted_tokens.push_back(correction_token);
                  }
             }
             
             if (!conversation_id.empty() && cache_manager_) {
                 cache_manager_->Put(conversation_id, std::move(kv_cache_->values), 0);
             }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[ModelLoader::VerifyDraft] Error: " << e.what() << std::endl;
        return {};
    }
    
    return accepted_tokens;
}

void ModelLoader::ClearCache(const std::string& conversation_id) {
    std::lock_guard<std::mutex> lock(kv_mutex_);
    if (!cache_manager_) return;
    
    if (conversation_id.empty()) {
        cache_manager_->Clear();
    } else {
        cache_manager_->Remove(conversation_id);
    }
}

void ModelLoader::PrintCacheStats() const {
    std::lock_guard<std::mutex> lock(kv_mutex_);
    if (!cache_manager_) return;
    
    auto stats = cache_manager_->GetStats();
    std::cerr << "\n=== KV-Cache Statistics ===" << std::endl;
    std::cerr << "Total Entries: " << stats.total_entries << std::endl;
    std::cerr << "Memory Usage: " << (stats.total_memory_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cerr << "Hit Rate: " << (stats.hit_rate() * 100.0f) << "%" << std::endl;
    std::cerr << "==========================\n" << std::endl;
}

int64_t ModelLoader::SampleToken(float* logits, int64_t vocab_size) {
    if (temperature_ > 0.0f) {
        for (int64_t i = 0; i < vocab_size; ++i) logits[i] /= temperature_;
    }
    
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) probs[i] /= sum_exp;
    
    if (top_p_ < 1.0f && top_p_ > 0.0f) {
        std::vector<std::pair<float, int64_t>> prob_indices(vocab_size);
        for (int i = 0; i < vocab_size; ++i) prob_indices[i] = {probs[i], i};
        std::sort(prob_indices.begin(), prob_indices.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                 
        float cum_prob = 0.0f;
        int64_t cutoff_index = vocab_size - 1;
        for (int64_t i = 0; i < vocab_size; ++i) {
            cum_prob += prob_indices[i].first;
            if (cum_prob > top_p_) { cutoff_index = i; break; }
        }
        
        std::vector<float> filtered_probs;
        std::vector<int64_t> filtered_indices;
        float new_sum = 0.0f;
        for (int64_t i = 0; i <= cutoff_index; ++i) {
            filtered_probs.push_back(prob_indices[i].first);
            filtered_indices.push_back(prob_indices[i].second);
            new_sum += prob_indices[i].first;
        }
        
        std::uniform_real_distribution<> dis(0.0, new_sum);
        float r;
        {
            std::lock_guard<std::mutex> lock(gen_mutex_);
            r = dis(gen_);
        }
        float acc = 0.0f;
        for (size_t i = 0; i < filtered_probs.size(); ++i) {
            acc += filtered_probs[i];
            if (acc >= r) return filtered_indices[i];
        }
        return filtered_indices.back();
    }
    
    return std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
}

void ModelLoader::Cancel(const std::string& conversation_id) {
    std::lock_guard<std::mutex> lock(cancel_mutex_);
    if (conversation_id.empty()) {
        for (auto& pair : cancel_flags_) {
            pair.second = true;
        }
    } else {
        cancel_flags_[conversation_id] = true;
    }
}

} // namespace NPCInference
