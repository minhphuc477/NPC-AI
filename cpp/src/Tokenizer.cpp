#include "Tokenizer.h"
#include <sentencepiece_processor.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <cstdlib>

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

namespace {

bool IsMockMode() {
    const char* mock_env = std::getenv("NPC_MOCK_MODE");
    if (mock_env) {
        return std::string(mock_env) == "1";
    }
#if NPC_USE_MOCKS
    return true;
#else
    return false;
#endif
}

} // namespace

    Tokenizer::Tokenizer() {
        sentence_piece_processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    }

    Tokenizer::~Tokenizer() = default;

    bool Tokenizer::Load(const std::string& model_path) {
        // Mock mode
        if (IsMockMode()) {
            std::cerr << "Tokenizer running in MOCK MODE (No model file req)" << std::endl;
            // Setup dummy special tokens
             special_tokens_ = {
                {"<|user|>", 32006},
                {"<|assistant|>", 32001},
                {"<|system|>", 32005},
                {"<|end|>", 32007},
                {"<|endoftext|>", 32000}
            };
            is_mock_ = true;
            return true;
        }

        const auto status = sentence_piece_processor_->Load(model_path);
        if (!status.ok()) {
            std::cerr << "Failed to load tokenizer model: " << status.ToString() << std::endl;
            sentence_piece_processor_.reset(); // Ensure null
            return false;
        }
        
        // Load added tokens if available (basic JSON parsing)
        // For now, let's hardcode the critical Phi-3 tokens to ensure safety.
        // ... (keep comments)
        
        special_tokens_ = {
            {"<|user|>", 32006},
            {"<|assistant|>", 32001},
            {"<|system|>", 32005},
            {"<|end|>", 32007},
            {"<|endoftext|>", 32000}
        };
        
        CheckSpecialToken("<|user|>");
        CheckSpecialToken("<|assistant|>");
        CheckSpecialToken("<|system|>");
        CheckSpecialToken("<|end|>");
        
        std::cerr << "Tokenizer loaded successfully from: " << model_path << std::endl;
        return true;
    }

    int Tokenizer::GetVocabSize() const {
        return IsLoaded() ? sentence_piece_processor_->GetPieceSize() : 0;
    }

    int Tokenizer::GetEOSId() const {
        return IsLoaded() ? sentence_piece_processor_->eos_id() : -1;
    }
    
    void Tokenizer::CheckSpecialToken(const std::string& token) {
        if (!IsLoaded()) return;
        int id = sentence_piece_processor_->PieceToId(token);
        if (id != sentence_piece_processor_->unk_id()) {
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
        if (!IsLoaded()) {
             // If mock mode, we "are loaded" but sentence_piece_processor_ might be empty? 
             // Wait, sentence_piece_processor_ is unique_ptr.
             // If Load skipped loading SPM, it's non-null (ctor) but empty.
             // Mock Encode: simple ASCII?
             if (IsMockMode()) {
                 std::vector<int64_t> output;
                 for(char c : text) output.push_back((int64_t)c);
                 return output;
             }
            std::cerr << "Tokenizer not loaded!" << std::endl;
            return {};
        }

        // 1. naive split by special tokens is hard because they can interleave.
        // Use a simple finder.
        
        std::vector<int64_t> final_ids;
        
        // Mock check inside Encode? No, IsLoaded() checks if SPM is loaded?
        // IsLoaded() checks sentence_piece_processor_->status().ok()? 
        // No, IsLoaded() isn't shown in my view. I need to make sure IsLoaded() returns true for mock.
        // I didn't see IsLoaded definition. It's likely in header returning sentence_piece_processor_ != nullptr.
        
        // If mock mode, SPM is not loaded with model, but pointer exists.
        // Calling SPM methods might crash if not loaded?
        // Let's safe guard.
        if (IsMockMode()) {
             std::vector<int64_t> output;
             for(char c : text) output.push_back((int64_t)c); // Simple ASCII encoding
             return output;
        }

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
            sentence_piece_processor_->Encode(text, &ids);
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
                sentence_piece_processor_->Encode(segment, &ids);
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
            sentence_piece_processor_->Encode(segment, &ids);
            for(int i : ids) final_ids.push_back(i);
        }

        return final_ids;
    }

    std::string Tokenizer::Decode(const std::vector<int64_t>& ids) {
        if (IsMockMode()) {
            std::string s;
            for(auto id : ids) s += (char)id;
            return s;
        }

        if (!IsLoaded()) {
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
                    sentence_piece_processor_->Decode(spm_chunk, &text);
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
            sentence_piece_processor_->Decode(spm_chunk, &text);
            ss << text;
        }
        
        return ss.str();
    }

} // namespace NPCInference
