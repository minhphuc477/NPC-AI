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
        virtual bool Load(const std::string& model_path);

        /**
         * Encode text to token IDs
         * @param text Input text
         * @return Vector of token IDs
         */
        virtual std::vector<int64_t> Encode(const std::string& text);

        /**
         * Decode token IDs to text
         * @param ids Vector of token IDs
         * @return Decoded text
         */
        virtual std::string Decode(const std::vector<int64_t>& ids);

        /**
         * Get the vocabulary size.
         * @return The size of the vocabulary.
         */
        virtual int GetVocabSize() const;

        /**
         * Get the End-Of-Sentence (EOS) token ID.
         * @return The EOS token ID.
         */
        virtual int GetEOSId() const;

        /**
         * Check if tokenizer is loaded
         */
        virtual bool IsLoaded() const { return is_mock_ || sentence_piece_processor_ != nullptr; }

    private:
        void CheckSpecialToken(const std::string& token);

        std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor_;
        std::map<std::string, int64_t> special_tokens_;
        bool is_mock_ = false;
    };

} // namespace NPCInference
