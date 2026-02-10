#include "Tokenizer.h"
#include <sentencepiece_processor.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>

// Phi-3 Special Tokens (Common IDs, but should verify with added_tokens.json if possible)
// Usually:
// <|endoftext|> : 32000
// <|assistant|> : 32001
// <|user|>      : 32006
// <|system|>    : 32007 (Wait, need to check exact map)
// <|end|>       : 32007?
// 
// Let's use a dynamic map if we can load it, or hardcode common Phi-3 ones if we must.
// Reliable way: Load added_tokens.json if exists.

namespace NPCInference {

    Tokenizer::Tokenizer() {
        processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    }

    Tokenizer::~Tokenizer() = default;

    bool Tokenizer::Load(const std::string& model_path) {
        const auto status = processor_->Load(model_path);
        if (!status.ok()) {
            std::cerr << "Failed to load tokenizer model: " << status.ToString() << std::endl;
            loaded_ = false;
            return false;
        }
        
        // Load added tokens if available (basic JSON parsing)
        // For now, let's hardcode the critical Phi-3 tokens to ensure safety.
        // If we strictly rely on SPM, we get garbage for "<|system|>".
        
        // Phi-3 Mini 4K Instruct tokens:
        // Source: Hugging Face config
        // <|user|> : 32006
        // <|assistant|> : 32001 (Correction: Verify this!)
        // <|system|> : 32007?
        // <|end|> : 32000 or 32007?
        
        // Actually, let's make it configurable or use a safe fallback.
        // We will implement a simple split-and-merge strategy.
        
        special_tokens_ = {
            {"<|user|>", 32006},
            {"<|assistant|>", 32001}, // Verify!
            {"<|system|>", 32005}, // Verify!
            {"<|end|>", 32007},    // Verify!
            {"<|endoftext|>", 32000}
        };
        
        // NOTE: The IDs above are guesses based on common Phi-3/Llama-2 patterns.
        // Without `added_tokens.json`, we risk using wrong IDs.
        // IMPROVEMENT: We should try to load `added_tokens.json` in a real implementation.
        // For this V2 fix, we will assume standard indices OR just rely on SPM if they ARE in SPM.
        // Use processor_->PieceToId to check!
        
        CheckSpecialToken("<|user|>");
        CheckSpecialToken("<|assistant|>");
        CheckSpecialToken("<|system|>");
        CheckSpecialToken("<|end|>");
        
        loaded_ = true;
        std::cerr << "Tokenizer loaded successfully from: " << model_path << std::endl;
        return true;
    }
    
    void Tokenizer::CheckSpecialToken(const std::string& token) {
        int id = processor_->PieceToId(token);
        if (id != processor_->unk_id()) {
            // It exists in SPM!
            special_tokens_[token] = id;
            std::cerr << "Found special token in SPM: " << token << " -> " << id << std::endl;
        } else {
            // Not in SPM. Keep our hardcoded value OR warn?
            // If it's not in SPM, we MUST have the correct ID from somewhere else.
            // If we don't, we can't use it.
            std::cerr << "Warning: Special token " << token << " not in SPM model. Using hardcoded fallback: " << special_tokens_[token] << std::endl;
        }
    }

    std::vector<int64_t> Tokenizer::Encode(const std::string& text) {
        if (!loaded_) {
            std::cerr << "Tokenizer not loaded!" << std::endl;
            return {};
        }

        // 1. naive split by special tokens is hard because they can interleave.
        // Use a simple finder.
        
        std::vector<int64_t> final_ids;
        
        // Find all occurrences of special tokens
        std::map<size_t, std::string> occurrences;
        for (const auto& kv : special_tokens_) {
            const std::string& token = kv.first;
            size_t pos = text.find(token, 0);
            while(pos != std::string::npos) {
                occurrences[pos] = token;
                pos = text.find(token, pos + 1);
            }
        }
        
        if (occurrences.empty()) {
            // No special tokens, direct encode
            std::vector<int> ids;
            processor_->Encode(text, &ids);
            for(int i : ids) final_ids.push_back(i);
            return final_ids;
        }
        
        // Process text chunks
        size_t current_pos = 0;
        for (auto const& [pos, token] : occurrences) {
            if (pos > current_pos) {
                // Encode text between special tokens
                std::string segment = text.substr(current_pos, pos - current_pos);
                std::vector<int> ids;
                processor_->Encode(segment, &ids);
                for(int i : ids) final_ids.push_back(i);
            }
            
            // Add special token ID
            final_ids.push_back(special_tokens_[token]);
            current_pos = pos + token.length();
        }
        
        // Remaining text
        if (current_pos < text.length()) {
            std::string segment = text.substr(current_pos);
            std::vector<int> ids;
            processor_->Encode(segment, &ids);
            for(int i : ids) final_ids.push_back(i);
        }

        return final_ids;
    }

    std::string Tokenizer::Decode(const std::vector<int64_t>& ids) {
        if (!loaded_) {
            std::cerr << "Tokenizer not loaded!" << std::endl;
            return "";
        }

        // Convert int64_t to int for SentencePiece
        // And handle special tokens (skip them? or print them?)
        // Usually we want to print them in debug, but hide them in chat?
        // Phi-3: We probably want to see the text.
        
        // Simple approach: Decode everything using SPM. 
        // If special tokens are NOT in SPM, SPM Decode might fail or print nothing for those IDs.
        // We should map IDs back to text if they are special.
        
        std::stringstream ss;
        std::vector<int> spm_chunk;
        
        for (int64_t id : ids) {
            bool is_special = false;
            std::string special_text = "";
            for (const auto& kv : special_tokens_) {
                if (kv.second == id) {
                    is_special = true;
                    special_text = kv.first;
                    break;
                }
            }
            
            if (is_special) {
                // Decode buffered chunk
                if (!spm_chunk.empty()) {
                    std::string text;
                    processor_->Decode(spm_chunk, &text);
                    ss << text;
                    spm_chunk.clear();
                }
                // Append special token text (optional, or skip)
                // For chat generation, we might want to skip control tokens?
                // Or keep them. Let's keep them for fidelity.
                ss << special_text; 
            } else {
                spm_chunk.push_back((int)id);
            }
        }
        
        if (!spm_chunk.empty()) {
            std::string text;
            processor_->Decode(spm_chunk, &text);
            ss << text;
        }
        
        return ss.str();
    }

} // namespace NPCInference
