#pragma once

#include <vector>
#include <string>
#include <stack>
#include <unordered_set>
#include <cstdint>

namespace NPCInference {

class Tokenizer;

class GrammarSampler {
public:
    enum class JsonState {
        WAITING_FOR_OPEN_BRACE,
        WAITING_FOR_KEY_QUOTE_START,
        INSIDE_KEY,
        WAITING_FOR_COLON,
        WAITING_FOR_VALUE_START,
        INSIDE_STRING_VALUE,
        INSIDE_NUMBER_VALUE,
        INSIDE_BOOLEAN_VALUE,
        INSIDE_NULL_VALUE,
        WAITING_FOR_ARRAY_START,
        INSIDE_ARRAY,
        WAITING_FOR_COMMA_OR_CLOSE,
        DONE
    };

    GrammarSampler(Tokenizer* tokenizer);
    ~GrammarSampler();

    void Reset();

    /**
     * Update the state machine based on the sampled token
     */
    void AcceptToken(int64_t token_id);

    /**
     * Modify logits to allow only valid tokens for the current state
     * @param logits Raw logits array
     * @param vocab_size Size of vocab
     */
    void FilterLogits(float* logits, int64_t vocab_size);

    /**
     * Validate generated JSON string
     * @param json_str JSON string to validate
     * @return true if valid JSON
     */
    bool ValidateJSON(const std::string& json_str);

    /**
     * Get current state (for debugging)
     */
    JsonState GetState() const { return state_; }

private:
    Tokenizer* tokenizer_;
    JsonState state_ = JsonState::WAITING_FOR_OPEN_BRACE;
    bool in_escape_seq_ = false;
    
    // Cache common token IDs
    int64_t id_brace_open_ = -1;
    int64_t id_brace_close_ = -1;
    int64_t id_bracket_open_ = -1;
    int64_t id_bracket_close_ = -1;
    int64_t id_quote_ = -1;
    int64_t id_colon_ = -1;
    int64_t id_comma_ = -1;
    
    // Value type tokens
    std::vector<int64_t> id_digits_;
    std::vector<int64_t> id_true_;
    std::vector<int64_t> id_false_;
    std::vector<int64_t> id_null_;
    
    // State tracking
    int brace_depth_ = 0;
    int bracket_depth_ = 0;
    std::string partial_json_;
    std::stack<JsonState> state_stack_;

    void ResolveTokenIds();
    bool IsDigitToken(int64_t token_id) const;
    bool IsBooleanToken(int64_t token_id) const;
    bool IsNullToken(int64_t token_id) const;
};

} // namespace NPCInference
