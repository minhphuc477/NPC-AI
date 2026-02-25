#include "ResponseController.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <sstream>
#include <unordered_set>

namespace NPCInference {

namespace {

const std::vector<std::string> kBlockedFragments = {
    "your response here",
    "solution to instruction",
    "instruction with the same difficulty",
    "temporal memories",
    "ambient awareness",
    "behaviortreestate",
};

const std::unordered_set<std::string> kIntentStopwords = {
    "i", "me", "my", "you", "your", "we", "us", "please", "can", "could", "would", "should",
    "let", "need", "want", "now", "to", "in", "into", "at", "the", "a", "an", "is", "are",
    "do", "does",
};

const std::unordered_set<std::string> kContextStopwords = {
    "you", "are", "the", "and", "with", "that", "this", "your", "from", "into", "about",
    "while", "who", "for", "can", "will", "game", "npc",
};

const std::vector<std::string> kPersonaTerms = {
    "strict", "fair", "brief", "suspicious", "talkative", "profit", "calm", "caring",
    "precise", "direct", "practical", "formal", "sources", "precision", "mysterious",
    "indirect", "procedural", "scholar", "healer", "guard", "merchant",
};

std::string ToLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::string Trim(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }

    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(start, end - start);
}

std::string TrimQuotes(const std::string& text) {
    size_t start = 0;
    while (start < text.size() &&
           (std::isspace(static_cast<unsigned char>(text[start])) || text[start] == '"' ||
            text[start] == '\'')) {
        ++start;
    }

    size_t end = text.size();
    while (end > start &&
           (std::isspace(static_cast<unsigned char>(text[end - 1])) || text[end - 1] == '"' ||
            text[end - 1] == '\'')) {
        --end;
    }
    return text.substr(start, end - start);
}

bool StartsWith(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() &&
           std::equal(prefix.begin(), prefix.end(), text.begin());
}

bool ContainsAny(const std::string& lowered_text, const std::vector<std::string>& fragments) {
    for (const auto& fragment : fragments) {
        if (lowered_text.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> Tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    const std::string lowered = ToLower(text);

    for (char c : lowered) {
        const bool is_token_char =
            std::isalnum(static_cast<unsigned char>(c)) || c == '\'';
        if (is_token_char) {
            current.push_back(c);
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

std::string Join(const std::vector<std::string>& parts, const std::string& delimiter) {
    if (parts.empty()) {
        return "";
    }
    std::ostringstream out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out << delimiter;
        }
        out << parts[i];
    }
    return out.str();
}

std::vector<std::string> SplitByDelimiters(const std::string& text, const std::string& delimiters) {
    std::vector<std::string> out;
    std::string current;
    for (char c : text) {
        if (delimiters.find(c) != std::string::npos) {
            const std::string trimmed = Trim(current);
            if (!trimmed.empty()) {
                out.push_back(trimmed);
            }
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    const std::string trimmed = Trim(current);
    if (!trimmed.empty()) {
        out.push_back(trimmed);
    }
    return out;
}

std::vector<std::string> SplitLines(const std::string& text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (char c : text) {
        if (c == '\r') {
            normalized.push_back('\n');
        } else {
            normalized.push_back(c);
        }
    }
    return SplitByDelimiters(normalized, "\n");
}

bool IsLabelChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == ' ' || c == '(' ||
           c == ')' || c == '-';
}

std::string RemoveLeadingSpeakerLabel(const std::string& text) {
    std::string output = Trim(text);
    if (output.empty()) {
        return output;
    }

    if (output.front() == '[') {
        const size_t close = output.find(']');
        if (close != std::string::npos && close < 64) {
            output = Trim(output.substr(close + 1));
        }
    }

    const size_t colon = output.find(':');
    if (colon != std::string::npos && colon > 0 && colon <= 48) {
        bool valid_label = true;
        for (size_t i = 0; i < colon; ++i) {
            if (!IsLabelChar(output[i])) {
                valid_label = false;
                break;
            }
        }
        if (valid_label) {
            output = Trim(output.substr(colon + 1));
        }
    }
    return output;
}

std::vector<std::string> SplitSentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;
    for (char c : text) {
        current.push_back(c);
        if (c == '.' || c == '!' || c == '?') {
            const std::string trimmed = Trim(current);
            if (!trimmed.empty()) {
                sentences.push_back(trimmed);
            }
            current.clear();
        }
    }
    const std::string tail = Trim(current);
    if (!tail.empty()) {
        sentences.push_back(tail);
    }
    return sentences;
}

float KeywordCoverage(const std::string& text, const std::vector<std::string>& keywords) {
    std::vector<std::string> cleaned_keywords;
    cleaned_keywords.reserve(keywords.size());
    for (const auto& kw : keywords) {
        const std::string cleaned = Trim(ToLower(kw));
        if (!cleaned.empty()) {
            cleaned_keywords.push_back(cleaned);
        }
    }
    if (cleaned_keywords.empty()) {
        return 0.0f;
    }

    const std::string lowered = ToLower(text);
    const auto text_tokens = Tokenize(text);
    const std::unordered_set<std::string> token_set(text_tokens.begin(), text_tokens.end());

    int hits = 0;
    for (const auto& keyword : cleaned_keywords) {
        if (lowered.find(keyword) != std::string::npos) {
            ++hits;
            continue;
        }
        const auto kw_tokens = Tokenize(keyword);
        if (kw_tokens.empty()) {
            continue;
        }
        bool all_tokens_present = true;
        for (const auto& token : kw_tokens) {
            if (token_set.find(token) == token_set.end()) {
                all_tokens_present = false;
                break;
            }
        }
        if (all_tokens_present) {
            ++hits;
        }
    }

    return static_cast<float>(hits) / static_cast<float>(cleaned_keywords.size());
}

std::vector<std::string> DedupeKeywords(const std::vector<std::string>& candidates, size_t max_items) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    out.reserve(std::min(candidates.size(), max_items));

    for (const auto& raw : candidates) {
        const std::string normalized = Trim(ToLower(raw));
        if (normalized.empty()) {
            continue;
        }
        if (seen.insert(normalized).second) {
            out.push_back(normalized);
            if (out.size() >= max_items) {
                break;
            }
        }
    }
    return out;
}

std::string JsonValueToText(const nlohmann::json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "true" : "false";
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<long long>());
    }
    if (value.is_number_unsigned()) {
        return std::to_string(value.get<unsigned long long>());
    }
    if (value.is_number_float()) {
        return std::to_string(value.get<double>());
    }
    return value.dump();
}

void CollectContextFields(
    const nlohmann::json& node,
    int depth,
    std::vector<std::pair<std::string, std::string>>& fields
) {
    if (depth > 3) {
        return;
    }

    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            const auto& key = it.key();
            const auto& value = it.value();
            if (value.is_primitive()) {
                fields.emplace_back(key, JsonValueToText(value));
            } else {
                CollectContextFields(value, depth + 1, fields);
            }
        }
        return;
    }

    if (node.is_array()) {
        size_t seen = 0;
        for (const auto& item : node) {
            if (seen >= 4) {
                break;
            }
            if (item.is_primitive()) {
                fields.emplace_back("value", JsonValueToText(item));
            } else {
                CollectContextFields(item, depth + 1, fields);
            }
            ++seen;
        }
    }
}

std::pair<bool, std::string> NeedsRepair(
    const std::string& response,
    float context_cov,
    float persona_cov,
    const ResponseControlConfig& config
) {
    const std::string lowered = ToLower(response);
    if (response.empty()) {
        return {true, "empty_response"};
    }
    if (Tokenize(response).size() < 8) {
        return {true, "too_short"};
    }
    if (ContainsAny(lowered, kBlockedFragments)) {
        return {true, "meta_artifact"};
    }
    if (context_cov < config.min_context_coverage) {
        return {true, "low_context_coverage"};
    }
    if (persona_cov < config.min_persona_coverage) {
        return {true, "low_persona_coverage"};
    }
    return {false, ""};
}

float CandidateLengthScore(const std::string& response) {
    const float wc = static_cast<float>(Tokenize(response).size());
    if (wc <= 0.0f) {
        return 0.0f;
    }
    const float target = 34.0f;
    const float spread = 28.0f;
    const float score = 1.0f - std::abs(wc - target) / spread;
    return std::max(0.0f, score);
}

float CandidateDiversityScore(const std::string& response) {
    const auto words = Tokenize(response);
    const size_t wc = words.size();
    if (wc == 0) {
        return 0.0f;
    }

    std::unordered_set<std::string> uniq(words.begin(), words.end());
    const float distinct1 = static_cast<float>(uniq.size()) / static_cast<float>(wc);

    float repetition = 0.0f;
    if (wc >= 3) {
        std::map<std::string, int> gram_counts;
        int grams_total = 0;
        for (size_t i = 0; i + 2 < wc; ++i) {
            const std::string gram = words[i] + "|" + words[i + 1] + "|" + words[i + 2];
            ++gram_counts[gram];
            ++grams_total;
        }
        int repeats = 0;
        for (const auto& kv : gram_counts) {
            if (kv.second > 1) {
                repeats += (kv.second - 1);
            }
        }
        if (grams_total > 0) {
            repetition = static_cast<float>(repeats) / static_cast<float>(grams_total);
        }
    }

    const float score = 0.6f * distinct1 + 0.4f * (1.0f - repetition);
    return std::max(0.0f, std::min(1.0f, score));
}

float CandidateScore(
    const std::string& response,
    const std::vector<std::string>& context_keywords,
    const std::vector<std::string>& persona_keywords
) {
    if (response.empty()) {
        return -1.0f;
    }
    if (ContainsAny(ToLower(response), kBlockedFragments)) {
        return -1.0f;
    }
    const float context_cov = KeywordCoverage(response, context_keywords);
    const float persona_cov = KeywordCoverage(response, persona_keywords);
    const float length_score = CandidateLengthScore(response);
    const float diversity_score = CandidateDiversityScore(response);
    return 0.50f * context_cov + 0.30f * persona_cov + 0.10f * length_score + 0.10f * diversity_score;
}

std::vector<float> RewriteTemperatures(const ResponseControlConfig& config) {
    const int n = std::max(1, config.rewrite_candidates);
    const float base = std::min(1.5f, std::max(0.05f, config.rewrite_temperature));
    const float step = std::max(0.01f, config.rewrite_temperature_step);

    std::vector<float> temps;
    temps.reserve(static_cast<size_t>(n));
    temps.push_back(base);

    int radius = 1;
    while (static_cast<int>(temps.size()) < n) {
        const float hi = std::min(1.5f, base + static_cast<float>(radius) * step);
        if (std::find(temps.begin(), temps.end(), hi) == temps.end()) {
            temps.push_back(hi);
        }
        if (static_cast<int>(temps.size()) >= n) {
            break;
        }
        const float lo = std::max(0.05f, base - static_cast<float>(radius) * step);
        if (std::find(temps.begin(), temps.end(), lo) == temps.end()) {
            temps.push_back(lo);
        }
        ++radius;
    }

    if (static_cast<int>(temps.size()) > n) {
        temps.resize(static_cast<size_t>(n));
    }
    return temps;
}

std::map<std::string, std::string> ParseDynamicContext(const std::string& dynamic_context) {
    std::map<std::string, std::string> out;
    for (const auto& part : SplitByDelimiters(dynamic_context, ";")) {
        const std::string chunk = Trim(part);
        if (chunk.empty()) {
            continue;
        }

        const size_t eq = chunk.find('=');
        const size_t colon = chunk.find(':');
        size_t pos = std::string::npos;
        if (eq != std::string::npos) {
            pos = eq;
        } else if (colon != std::string::npos) {
            pos = colon;
        }
        if (pos == std::string::npos) {
            continue;
        }

        const std::string key = Trim(ToLower(chunk.substr(0, pos)));
        const std::string value = Trim(chunk.substr(pos + 1));
        if (!key.empty() && !value.empty()) {
            out[key] = value;
        }
    }
    return out;
}

std::vector<std::string> ContextDetailPhrases(const std::string& dynamic_context) {
    const auto kv = ParseDynamicContext(dynamic_context);
    std::vector<std::string> details;

    auto location_it = kv.find("location");
    if (location_it != kv.end()) {
        details.push_back("at " + location_it->second);
    }

    auto state_it = kv.find("behaviortreestate");
    if (state_it == kv.end()) {
        state_it = kv.find("behavior_state");
    }
    if (state_it != kv.end()) {
        details.push_back("while on " + ToLower(state_it->second) + " duty");
    }

    auto nearby_it = kv.find("nearbyentity");
    if (nearby_it == kv.end()) {
        nearby_it = kv.find("nearby_entity");
    }
    if (nearby_it != kv.end()) {
        details.push_back("with " + ToLower(nearby_it->second) + " nearby");
    }

    auto event_it = kv.find("recentevent");
    if (event_it == kv.end()) {
        event_it = kv.find("recent_event");
    }
    if (event_it != kv.end()) {
        details.push_back("after " + ToLower(event_it->second));
    }

    if (details.empty()) {
        int collected = 0;
        for (const auto& [key, value] : kv) {
            std::string k = key;
            std::replace(k.begin(), k.end(), '_', ' ');
            details.push_back(k + " " + value);
            ++collected;
            if (collected >= 2) {
                break;
            }
        }
    }
    return details;
}

std::vector<std::string> ContextDetailSentences(const std::string& dynamic_context) {
    const auto phrases = ContextDetailPhrases(dynamic_context);
    if (phrases.empty()) {
        return {};
    }

    std::vector<std::string> sentences;
    if (phrases.size() == 1) {
        sentences.push_back("We are " + phrases[0] + ".");
        return sentences;
    }

    sentences.push_back("We are " + phrases[0] + " and " + phrases[1] + ".");
    for (size_t i = 2; i < phrases.size(); ++i) {
        sentences.push_back("Current conditions remain " + phrases[i] + ".");
    }
    return sentences;
}

std::string PersonaStyle(const std::string& persona) {
    const std::string lowered = ToLower(persona);
    auto has_any = [&](std::initializer_list<const char*> terms) {
        for (const char* t : terms) {
            if (lowered.find(t) != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    if (has_any({"strict", "guard", "captain", "procedural"})) {
        return "strict";
    }
    if (has_any({"merchant", "talkative", "trader"})) {
        return "talkative";
    }
    if (has_any({"healer", "calm", "caring", "medical"})) {
        return "calm";
    }
    if (has_any({"witch", "mysterious", "indirect"})) {
        return "mysterious";
    }
    if (has_any({"scholar", "formal", "precision"})) {
        return "formal";
    }
    return "neutral";
}

std::string IntentFragment(const std::string& player_input) {
    const auto tokens = Tokenize(player_input);
    if (tokens.empty()) {
        return "your request";
    }

    std::vector<std::string> content_tokens;
    for (const auto& t : tokens) {
        if (kIntentStopwords.find(t) == kIntentStopwords.end()) {
            content_tokens.push_back(t);
        }
    }
    if (!content_tokens.empty()) {
        if (content_tokens.size() > 4) {
            content_tokens.resize(4);
        }
        return Join(content_tokens, " ");
    }

    std::vector<std::string> fallback = tokens;
    if (fallback.size() > 5) {
        fallback.resize(5);
    }
    return Join(fallback, " ");
}

std::string GroundedFallbackResponse(
    const std::string& persona,
    const std::string& dynamic_context,
    const std::string& player_input
) {
    const auto detail_phrases = ContextDetailPhrases(dynamic_context);
    const std::string style = PersonaStyle(persona);
    const std::string intent = IntentFragment(player_input);

    std::string context_sentence;
    if (!detail_phrases.empty()) {
        if (detail_phrases.size() == 1) {
            context_sentence = "Here " + detail_phrases[0] + ",";
        } else {
            context_sentence =
                "Here " + detail_phrases[0] + " and " + detail_phrases[1] + ",";
        }
    } else {
        context_sentence = "Given the current situation,";
    }

    if (style == "strict") {
        return context_sentence +
               " I cannot approve " + intent +
               " until verification is complete. Follow protocol and I will move this forward.";
    }
    if (style == "talkative") {
        return context_sentence +
               " I can work with " + intent +
               ", but the terms must stay fair. Keep it honest and we can close this deal quickly.";
    }
    if (style == "calm") {
        return context_sentence +
               " we can handle " + intent +
               " safely, step by step. Stay steady and I will guide the next action.";
    }
    if (style == "mysterious") {
        return context_sentence +
               " your path around " + intent +
               " has a cost. Choose carefully, because timing matters as much as power.";
    }
    if (style == "formal") {
        return context_sentence +
               " the request regarding " + intent +
               " requires evidence before any conclusion. Provide verifiable details and I will continue.";
    }
    return context_sentence +
           " I can respond to " + intent +
           " within these conditions. Give one clear detail and I will proceed precisely.";
}

std::string BuildRewritePrompt(
    const std::string& persona,
    const std::string& dynamic_context,
    const std::string& player_input,
    const std::string& draft_response
) {
    std::vector<std::string> lines;
    lines.reserve(16);
    lines.push_back("Rewrite the NPC response to satisfy strict quality constraints.");
    lines.push_back("Persona: " + persona);
    lines.push_back("Dynamic game state:");
    for (const auto& sentence : ContextDetailSentences(dynamic_context)) {
        lines.push_back("- " + sentence);
    }
    lines.push_back("Player says: " + player_input);
    lines.push_back("Draft response: " + draft_response);
    lines.push_back("Constraints:");
    lines.push_back("- Keep 2-3 natural sentences.");
    lines.push_back("- Keep role-play tone consistent with persona.");
    lines.push_back("- Use at least two concrete dynamic game details.");
    lines.push_back("- Avoid repeated phrasing; prefer precise varied wording.");
    lines.push_back("- No labels, no metadata, no JSON, no analysis.");
    lines.push_back("Return only the rewritten NPC dialogue.");
    return Join(lines, "\n");
}

} // namespace

std::string ResponseController::SanitizeResponse(const std::string& text) {
    if (text.empty()) {
        return "";
    }

    std::vector<std::string> cleaned_lines;
    cleaned_lines.reserve(8);

    for (const auto& raw_line : SplitLines(text)) {
        std::string line = Trim(raw_line);
        if (line.empty()) {
            continue;
        }
        const std::string low = ToLower(line);

        if (StartsWith(low, "system persona:") || StartsWith(low, "persona:") ||
            StartsWith(low, "rules:") || StartsWith(low, "npc reply")) {
            continue;
        }
        if (StartsWith(low, "[your response here]")) {
            continue;
        }
        if (StartsWith(low, "**solution")) {
            continue;
        }
        if (StartsWith(low, "as elara") && low.find("respond") != std::string::npos) {
            continue;
        }
        if (StartsWith(low, "elara") && line.find('=') != std::string::npos) {
            continue;
        }
        if (ContainsAny(low, kBlockedFragments)) {
            continue;
        }
        cleaned_lines.push_back(line);
    }

    std::string merged = Trim(Join(cleaned_lines, " "));
    if (merged.empty()) {
        return "";
    }

    merged = RemoveLeadingSpeakerLabel(merged);
    merged = TrimQuotes(merged);
    if (merged.empty()) {
        return "";
    }

    if (ContainsAny(ToLower(merged), kBlockedFragments)) {
        return "";
    }

    auto sentences = SplitSentences(merged);
    if (sentences.empty()) {
        return "";
    }
    if (sentences.size() > 3) {
        sentences.resize(3);
    }
    return Trim(Join(sentences, " "));
}

std::vector<std::string> ResponseController::ExtractContextKeywords(
    const nlohmann::json& context,
    size_t max_items
) {
    std::vector<std::pair<std::string, std::string>> fields;
    fields.reserve(64);
    CollectContextFields(context, 0, fields);

    std::vector<std::string> candidates;
    candidates.reserve(64);

    for (const auto& [raw_key, raw_value] : fields) {
        std::string key_text = ToLower(raw_key);
        std::replace(key_text.begin(), key_text.end(), '_', ' ');
        key_text = Trim(key_text);
        if (!key_text.empty() && kContextStopwords.find(key_text) == kContextStopwords.end()) {
            candidates.push_back(key_text);
        }

        const std::string value_text = ToLower(raw_value);
        for (const auto& fragment : SplitByDelimiters(value_text, ";,.")) {
            auto fragment_tokens = Tokenize(fragment);
            std::vector<std::string> filtered_tokens;
            filtered_tokens.reserve(fragment_tokens.size());
            for (const auto& token : fragment_tokens) {
                if (kContextStopwords.find(token) == kContextStopwords.end()) {
                    filtered_tokens.push_back(token);
                }
            }
            if (filtered_tokens.empty() || filtered_tokens.size() > 6) {
                continue;
            }
            candidates.push_back(Join(filtered_tokens, " "));
        }
    }

    return DedupeKeywords(candidates, max_items);
}

std::vector<std::string> ResponseController::ExtractPersonaKeywords(
    const std::string& persona,
    size_t max_items
) {
    const std::string lowered = ToLower(persona);
    std::vector<std::string> candidates;
    candidates.reserve(24);

    for (const auto& term : kPersonaTerms) {
        if (lowered.find(term) != std::string::npos) {
            candidates.push_back(term);
        }
    }

    for (const auto& chunk : SplitByDelimiters(lowered, ",:;")) {
        auto tokens = Tokenize(chunk);
        std::vector<std::string> filtered;
        filtered.reserve(tokens.size());
        for (const auto& token : tokens) {
            if (kContextStopwords.find(token) == kContextStopwords.end()) {
                filtered.push_back(token);
            }
        }
        if (!filtered.empty() && filtered.size() <= 4) {
            candidates.push_back(Join(filtered, " "));
        }
    }

    return DedupeKeywords(candidates, max_items);
}

std::string ResponseController::BuildDynamicContext(const nlohmann::json& context) {
    if (!context.is_object()) {
        return "";
    }

    std::vector<std::string> parts;
    parts.reserve(10);

    auto add_part = [&](const std::string& key, const nlohmann::json& value) {
        if (value.is_null()) {
            return;
        }
        const std::string text = Trim(JsonValueToText(value));
        if (!text.empty()) {
            parts.push_back(key + "=" + text);
        }
    };

    const std::vector<std::string> priority_keys = {
        "location", "behavior_state", "behaviortreestate", "mood_state", "health_state",
        "recent_event", "nearby_entity", "nearbyentity", "time_of_day", "current_action",
        "npc_id", "player_id",
    };

    for (const auto& key : priority_keys) {
        if (context.contains(key) && context[key].is_primitive()) {
            add_part(key, context[key]);
        }
    }

    if (context.contains("ambient_awareness") && context["ambient_awareness"].is_object()) {
        const auto& awareness = context["ambient_awareness"];
        if (awareness.contains("current_events") && awareness["current_events"].is_array() &&
            !awareness["current_events"].empty() && awareness["current_events"][0].is_object()) {
            const auto& event = awareness["current_events"][0];
            if (event.contains("description") && event["description"].is_primitive()) {
                add_part("recent_event", event["description"]);
            }
            if (event.contains("location") && event["location"].is_primitive()) {
                add_part("location", event["location"]);
            }
        }
    }

    if (parts.empty()) {
        for (auto it = context.begin(); it != context.end(); ++it) {
            if (it.value().is_primitive()) {
                add_part(it.key(), it.value());
                if (parts.size() >= 4) {
                    break;
                }
            }
        }
    }

    return Join(parts, "; ");
}

ResponseControlResult ResponseController::ControlResponse(
    const std::string& raw_response,
    const std::string& persona,
    const std::string& dynamic_context,
    const std::string& player_input,
    const std::vector<std::string>& context_keywords,
    const std::vector<std::string>& persona_keywords,
    const ResponseControlConfig& config,
    const RewriteFn& rewrite_fn
) {
    const std::string cleaned = SanitizeResponse(raw_response);
    const float context_cov = KeywordCoverage(cleaned, context_keywords);
    const float persona_cov = KeywordCoverage(cleaned, persona_keywords);

    auto [repair_needed, reason] = NeedsRepair(cleaned, context_cov, persona_cov, config);
    if (!repair_needed) {
        return ResponseControlResult{
            cleaned,
            "raw",
            context_cov,
            persona_cov,
            false,
            "",
        };
    }

    struct Candidate {
        std::string text;
        float context_cov = 0.0f;
        float persona_cov = 0.0f;
        float score = -1.0f;
    };

    Candidate best_passing;
    bool has_best_passing = false;
    Candidate best_candidate;
    bool has_best_candidate = false;

    if (config.enable_rewrite && rewrite_fn) {
        const std::string rewrite_prompt =
            BuildRewritePrompt(persona, dynamic_context, player_input, cleaned.empty() ? raw_response : cleaned);
        for (float temp : RewriteTemperatures(config)) {
            const std::string rewritten_raw =
                rewrite_fn(rewrite_prompt, config.rewrite_max_tokens, temp);
            const std::string rewritten = SanitizeResponse(rewritten_raw);
            if (rewritten.empty()) {
                continue;
            }

            const float rewritten_context_cov = KeywordCoverage(rewritten, context_keywords);
            const float rewritten_persona_cov = KeywordCoverage(rewritten, persona_keywords);
            const float rewritten_score = CandidateScore(rewritten, context_keywords, persona_keywords);

            Candidate candidate{
                rewritten,
                rewritten_context_cov,
                rewritten_persona_cov,
                rewritten_score,
            };

            if (!has_best_candidate || candidate.score > best_candidate.score) {
                best_candidate = candidate;
                has_best_candidate = true;
            }

            auto [rewrite_needed, _rewrite_reason] =
                NeedsRepair(rewritten, rewritten_context_cov, rewritten_persona_cov, config);
            (void)_rewrite_reason;
            if (!rewrite_needed) {
                if (!has_best_passing || candidate.score > best_passing.score) {
                    best_passing = candidate;
                    has_best_passing = true;
                }
            }
        }
    }

    if (has_best_passing) {
        return ResponseControlResult{
            best_passing.text,
            "rewritten",
            best_passing.context_cov,
            best_passing.persona_cov,
            true,
            reason,
        };
    }

    std::string fallback = GroundedFallbackResponse(persona, dynamic_context, player_input);
    fallback = SanitizeResponse(fallback);
    const float fallback_context_cov = KeywordCoverage(fallback, context_keywords);
    const float fallback_persona_cov = KeywordCoverage(fallback, persona_keywords);

    if (config.allow_best_effort_rewrite && has_best_candidate) {
        const float fallback_score = CandidateScore(fallback, context_keywords, persona_keywords);
        if (best_candidate.score >= (fallback_score + 0.05f)) {
            return ResponseControlResult{
                best_candidate.text,
                "rewritten_best_effort",
                best_candidate.context_cov,
                best_candidate.persona_cov,
                true,
                reason,
            };
        }
    }

    if (fallback.empty()) {
        fallback = "I need a clearer request before I can respond in character.";
    }
    return ResponseControlResult{
        fallback,
        "fallback",
        fallback_context_cov,
        fallback_persona_cov,
        true,
        reason,
    };
}

} // namespace NPCInference
