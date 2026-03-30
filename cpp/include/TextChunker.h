#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <regex>

namespace NPCInference {

class TextChunker {
public:
    struct Config {
        int chunk_size = 500;
        int chunk_overlap = 50;
        std::vector<std::string> separators = {"\n\n", "\n", ". ", " ", ""};
    };

    static std::vector<std::string> SplitText(const std::string& text, const Config& config = Config()) {
        std::vector<std::string> final_chunks;
        std::vector<std::string> splits = SplitTextRecursive(text, config.separators, config.chunk_size);
        
        // Merge splits into chunks
        std::string current_chunk;
        
        for (const auto& split : splits) {
            if (current_chunk.length() + split.length() > config.chunk_size) {
                if (!current_chunk.empty()) {
                    final_chunks.push_back(current_chunk);
                    
                    // Handle overlap (simple approach: keep last N chars)
                    if (config.chunk_overlap > 0 && current_chunk.length() > config.chunk_overlap) {
                        current_chunk = current_chunk.substr(current_chunk.length() - config.chunk_overlap);
                    } else {
                        current_chunk = "";
                    }
                }
            }
            current_chunk += split;
        }
        
        if (!current_chunk.empty()) {
            final_chunks.push_back(current_chunk);
        }
        
        return final_chunks;
    }

private:
    static std::vector<std::string> SplitTextRecursive(const std::string& text, const std::vector<std::string>& separators, int chunk_size) {
        std::vector<std::string> final_chunks;
        std::string separator = separators.back();
        std::vector<std::string> new_separators;
        
        // Find the best separator to use
        bool found = false;
        for (const auto& sep : separators) {
            if (sep == "") {
                separator = "";
                found = true;
                break;
            }
            if (text.find(sep) != std::string::npos) {
                separator = sep;
                found = true;
                // Next separators are the ones following this one
                bool start_collecting = false;
                for (const auto& s : separators) {
                    if (start_collecting) new_separators.push_back(s);
                    if (s == separator) start_collecting = true;
                }
                break;
            }
        }
        
        if (!found) {
            separator = separators.back(); // Use character splitting if nothing else matches
        }

        // Split by separator
        std::vector<std::string> splits;
        if (separator.empty()) {
            // Character split
            for (char c : text) splits.push_back(std::string(1, c));
        } else {
            size_t start = 0;
            size_t end = text.find(separator);
            while (end != std::string::npos) {
                splits.push_back(text.substr(start, end - start + separator.length())); // Include separator
                start = end + separator.length();
                end = text.find(separator, start);
            }
            splits.push_back(text.substr(start));
        }

        // Process splits
        std::vector<std::string> good_splits;
        for (const auto& s : splits) {
            if (s.length() < chunk_size || new_separators.empty()) {
                good_splits.push_back(s);
            } else {
                // Recurse
                std::vector<std::string> sub_splits = SplitTextRecursive(s, new_separators, chunk_size);
                good_splits.insert(good_splits.end(), sub_splits.begin(), sub_splits.end());
            }
        }

        return good_splits;
    }
};

} // namespace NPCInference
