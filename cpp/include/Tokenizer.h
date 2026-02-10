#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

namespace sentencepiece {
    class SentencePieceProcessor;
}

namespace NPCInference {

    /**
     * Tokenizer wrapper for SentencePiece
     * Handles encoding text to token IDs and decoding back to text
     */
    class Tokenizer {
    public:
        Tokenizer();
        ~Tokenizer();

        /**
         * Load tokenizer model from file
         * @param model_path Path to tokenizer.model file
         * @return true if loaded successfully
         */
        bool Load(const std::string& model_path);

        /**
         * Encode text to token IDs
         * @param text Input text
         * @return Vector of token IDs
         */
        std::vector<int64_t> Encode(const std::string& text);

        /**
         * Decode token IDs to text
         * @param ids Vector of token IDs
         * @return Decoded text
         */
        std::string Decode(const std::vector<int64_t>& ids);

        /**
         * Check if tokenizer is loaded
         */
        bool IsLoaded() const { return loaded_; }

    private:
        void CheckSpecialToken(const std::string& token);

        std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
        bool loaded_ = false;
        std::map<std::string, int64_t> special_tokens_;
    };

} // namespace NPCInference
