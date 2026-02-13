#include "GrammarSampler.h"
#include "Tokenizer.h"
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stack>

namespace NPCInference {

GrammarSampler::GrammarSampler(Tokenizer* tokenizer) : tokenizer_(tokenizer) {
    ResolveTokenIds();
}

GrammarSampler::~GrammarSampler() = default;

void GrammarSampler::Reset() {
    state_ = JsonState::WAITING_FOR_OPEN_BRACE;
    in_escape_seq_ = false;
    brace_depth_ = 0;
    bracket_depth_ = 0;
    partial_json_.clear();
}

void GrammarSampler::ResolveTokenIds() {
    if (!tokenizer_) return;
    
    auto safe_encode = [&](const std::string& text) -> int64_t {
        auto ids = tokenizer_->Encode(text);
        return ids.empty() ? -1 : ids[0];
    };

    id_brace_open_ = safe_encode("{");
    id_brace_close_ = safe_encode("}");
    id_bracket_open_ = safe_encode("[");
    id_bracket_close_ = safe_encode("]");
    id_quote_ = safe_encode("\"");
    id_colon_ = safe_encode(":");
    id_comma_ = safe_encode(",");
    
    id_digits_.clear();
    for (int i = 0; i <= 9; ++i) {
        int64_t id = safe_encode(std::to_string(i));
        if (id != -1) id_digits_.push_back(id);
    }
    
    auto add_word = [&](const std::string& word, std::vector<int64_t>& vec) {
        vec.clear();
        auto ids = tokenizer_->Encode(word);
        if (!ids.empty()) vec = ids;
    };
    
    add_word("true", id_true_);
    add_word("false", id_false_);
    add_word("null", id_null_);

    std::cerr << "GrammarSampler: Resolved token IDs - {:" << id_brace_open_ 
              << " }:" << id_brace_close_ 
              << " [:" << id_bracket_open_
              << " ]:" << id_bracket_close_
              << " \":" << id_quote_ << std::endl;
}

bool GrammarSampler::IsDigitToken(int64_t token_id) const {
    return std::find(id_digits_.begin(), id_digits_.end(), token_id) != id_digits_.end();
}

bool GrammarSampler::IsBooleanToken(int64_t token_id) const {
    return std::find(id_true_.begin(), id_true_.end(), token_id) != id_true_.end() ||
           std::find(id_false_.begin(), id_false_.end(), token_id) != id_false_.end();
}

bool GrammarSampler::IsNullToken(int64_t token_id) const {
    return std::find(id_null_.begin(), id_null_.end(), token_id) != id_null_.end();
}

void GrammarSampler::AcceptToken(int64_t token_id) {
    if (!tokenizer_) return;
    std::string text = tokenizer_->Decode({token_id});
    partial_json_ += text;
    
    // Update depth tracking
    for (char c : text) {
        if (c == '{') brace_depth_++;
        else if (c == '}') brace_depth_--;
        else if (c == '[') bracket_depth_++;
        else if (c == ']') bracket_depth_--;
    }
    
    // Enhanced state machine with better transitions
    for (char c : text) {
        switch (state_) {
            case JsonState::WAITING_FOR_OPEN_BRACE:
                if (c == '{') {
                    state_ = JsonState::WAITING_FOR_KEY_QUOTE_START;
                }
                break;
                
            case JsonState::WAITING_FOR_KEY_QUOTE_START:
                if (c == '\"') {
                    state_ = JsonState::INSIDE_KEY;
                } else if (c == '}') {
                    // Empty object
                    state_ = JsonState::DONE;
                }
                // Ignore whitespace
                break;
                
            case JsonState::INSIDE_KEY:
                if (c == '\"' && !in_escape_seq_) {
                    state_ = JsonState::WAITING_FOR_COLON;
                } else if (c == '\\\\') {
                    in_escape_seq_ = !in_escape_seq_;
                } else {
                    in_escape_seq_ = false;
                }
                break;
                
            case JsonState::WAITING_FOR_COLON:
                if (c == ':') {
                    state_ = JsonState::WAITING_FOR_VALUE_START;
                }
                break;
                
            case JsonState::WAITING_FOR_VALUE_START:
                if (c == '\"') {
                    state_ = JsonState::INSIDE_STRING_VALUE;
                } else if (c == '[') {
                    state_stack_.push(JsonState::INSIDE_ARRAY);
                    state_ = JsonState::WAITING_FOR_VALUE_START; // Arrays start with a value
                } else if (c == '{') {
                    state_stack_.push(JsonState::WAITING_FOR_COMMA_OR_CLOSE);
                    state_ = JsonState::WAITING_FOR_KEY_QUOTE_START;
                } else if (std::isdigit(c) || c == '-') {
                    state_ = JsonState::INSIDE_NUMBER_VALUE;
                } else if (c == 't' || c == 'f') {
                    state_ = JsonState::INSIDE_BOOLEAN_VALUE;
                } else if (c == 'n') {
                    state_ = JsonState::INSIDE_NULL_VALUE;
                }
                break;
                
            case JsonState::INSIDE_STRING_VALUE:
                if (c == '\"' && !in_escape_seq_) {
                    if (!state_stack_.empty()) {
                        state_ = state_stack_.top();
                        state_stack_.pop();
                    } else {
                        state_ = JsonState::WAITING_FOR_COMMA_OR_CLOSE;
                    }
                } else if (c == '\\') {
                    in_escape_seq_ = !in_escape_seq_;
                } else {
                    in_escape_seq_ = false;
                }
                break;
                
            case JsonState::INSIDE_NUMBER_VALUE:
            case JsonState::INSIDE_BOOLEAN_VALUE:
            case JsonState::INSIDE_NULL_VALUE:
                if (c == ',' || c == '}' || c == ']') {
                    // Re-evaluate this character in the popped state
                    if (!state_stack_.empty()) {
                        state_ = state_stack_.top();
                        state_stack_.pop();
                    } else {
                        state_ = JsonState::WAITING_FOR_COMMA_OR_CLOSE;
                    }
                    
                    if (c == ',') state_ = JsonState::WAITING_FOR_KEY_QUOTE_START;
                    else if (c == '}' && brace_depth_ == 0) state_ = JsonState::DONE;
                    else if (c == ']') state_ = JsonState::WAITING_FOR_COMMA_OR_CLOSE;
                }
                break;
                
            case JsonState::INSIDE_ARRAY:
                if (c == ']') {
                    if (!state_stack_.empty()) {
                        state_ = state_stack_.top();
                        state_stack_.pop();
                    } else {
                        state_ = JsonState::WAITING_FOR_COMMA_OR_CLOSE;
                    }
                } else if (c == ',') {
                    state_ = JsonState::WAITING_FOR_VALUE_START;
                }
                break;
                
            case JsonState::WAITING_FOR_COMMA_OR_CLOSE:
                if (c == ',') {
                    state_ = JsonState::WAITING_FOR_KEY_QUOTE_START;
                } else if (c == '}') {
                    if (brace_depth_ == 0) {
                        state_ = JsonState::DONE;
                    } else if (!state_stack_.empty()) {
                        state_ = state_stack_.top();
                        state_stack_.pop();
                    }
                }
                break;
                
            case JsonState::DONE:
                // Ignore any trailing tokens
                break;
        }
    }
}

void GrammarSampler::FilterLogits(float* logits, int64_t vocab_size) {
    float NEG_INF = -std::numeric_limits<float>::infinity();

    auto AllowOnly = [&](const std::vector<int64_t>& allowed) {
        std::vector<int64_t> valid_ids;
        for (int64_t id : allowed) {
            if (id >= 0 && id < vocab_size) valid_ids.push_back(id);
        }
        
        if (valid_ids.empty()) return;
        
        // Mask all
        for (int64_t i = 0; i < vocab_size; ++i) {
            logits[i] = NEG_INF;
        }
        // Unmask valid
        for (int64_t id : valid_ids) {
            logits[id] = 0.0f;
        }
    };

    if (id_brace_open_ == -1) ResolveTokenIds();

    switch (state_) {
        case JsonState::WAITING_FOR_OPEN_BRACE:
            AllowOnly({id_brace_open_});
            break;
            
        case JsonState::WAITING_FOR_KEY_QUOTE_START:
            // Allow quote for key or close brace for empty/end object
            AllowOnly({id_quote_, id_brace_close_});
            break;
            
        case JsonState::WAITING_FOR_COLON:
            AllowOnly({id_colon_});
            break;
            
        case JsonState::WAITING_FOR_VALUE_START: {
            // Allow string, number, boolean, null, array, or nested object
            std::vector<int64_t> allowed = {id_quote_, id_brace_open_, id_bracket_open_};
            // Add digits for numbers
            allowed.insert(allowed.end(), id_digits_.begin(), id_digits_.end());
            // Add boolean/null tokens
            allowed.insert(allowed.end(), id_true_.begin(), id_true_.end());
            allowed.insert(allowed.end(), id_false_.begin(), id_false_.end());
            allowed.insert(allowed.end(), id_null_.begin(), id_null_.end());
            AllowOnly(allowed);
            break;
        }
        
        case JsonState::WAITING_FOR_COMMA_OR_CLOSE:
            AllowOnly({id_comma_, id_brace_close_});
            break;
            
        case JsonState::INSIDE_KEY:
        case JsonState::INSIDE_STRING_VALUE:
        case JsonState::INSIDE_NUMBER_VALUE:
        case JsonState::INSIDE_BOOLEAN_VALUE:
        case JsonState::INSIDE_NULL_VALUE:
        case JsonState::INSIDE_ARRAY:
            // Allow most tokens, but could add specific restrictions
            // For now, let the model generate freely within these states
            break;
            
        case JsonState::DONE:
            // Generation complete, mask everything to force EOS
            for (int64_t i = 0; i < vocab_size; ++i) {
                logits[i] = NEG_INF;
            }
            break;
    }
}

bool GrammarSampler::ValidateJSON(const std::string& json_str) {
    try {
        auto j = nlohmann::json::parse(json_str);
        return true;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON validation failed: " << e.what() << std::endl;
        return false;
    }
}

} // namespace NPCInference
