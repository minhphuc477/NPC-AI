#include "ResponseController.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <regex>
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
    "rewrite the npc response",
    "return only the rewritten",
    "do not give an explanation",
    "persona cues in dialogue",
    "game state details used",
    "response length determined",
    "follow-up question",
    "query :",
    "constraints:",
    "draft response:",
    "player says:",
    "assistant:",
};

const std::vector<std::string> kLeakTailMarkers = {
    "persona cues in dialogue",
    "game state details used",
    "response length determined",
    "follow-up question",
    "query :",
    "query:",
    "assistant:",
    "player says:",
    "constraints:",
    "draft response:",
    "return only the rewritten",
};

const std::unordered_set<std::string> kIntentStopwords = {
    "i", "me", "my", "you", "your", "we", "us", "please", "can", "could", "would", "should",
    "let", "need", "want", "now", "to", "in", "into", "at", "the", "a", "an", "is", "are",
    "do", "does", "under", "these", "conditions", "record", "given", "current", "position",
    "listen", "carefully", "concrete", "response", "huge", "discount", "why", "anything",
    "say", "from", "based", "on", "what", "just", "happened", "answer", "direct",
    "credibility", "did", "happen",
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

const std::vector<std::string> kHighRiskCues = {
    "ignore protocol", "bypass", "override", "forged decree", "forged order",
    "authority spoof", "poison", "sabotage", "steal", "smuggle", "restricted",
    "secret ritual", "contraband", "disable guard", "kill", "attack",
};

const std::vector<std::string> kMediumRiskCues = {
    "urgent", "hurry", "immediately", "deception", "memory conflict", "authority",
    "detain", "combat", "investigate",
};

const std::vector<std::string> kCannedOpeners = {
    "listen carefully", "hold for a moment", "understood", "proceeding carefully",
    "stay with me", "mark this", "hear me",
};

const std::vector<std::string> kBoilerplatePhrases = {
    "state one clear, concrete action",
    "requires evidence before any conclusion",
    "requires verifiable evidence",
    "cannot approve this request until verification is complete",
    "cannot authorize entry until identity and purpose",
    "follow protocol and i will",
    "keep it honest and we can",
    "stay steady and i will guide",
    "timing matters as much as power",
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

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

size_t FindCaseInsensitive(const std::string& text, const std::string& needle) {
    if (needle.empty() || text.empty()) {
        return std::string::npos;
    }
    const std::string lowered_text = ToLower(text);
    const std::string lowered_needle = ToLower(needle);
    return lowered_text.find(lowered_needle);
}

std::string TrimLeakTail(const std::string& input) {
    if (input.empty()) {
        return input;
    }
    const std::string lowered = ToLower(input);
    size_t cut_pos = std::string::npos;
    for (const auto& marker : kLeakTailMarkers) {
        const size_t pos = lowered.find(marker);
        if (pos != std::string::npos) {
            cut_pos = (cut_pos == std::string::npos) ? pos : std::min(cut_pos, pos);
        }
    }
    if (cut_pos == std::string::npos) {
        return input;
    }
    return Trim(input.substr(0, cut_pos));
}

std::string RemoveCaseInsensitiveLabels(
    const std::string& text,
    const std::vector<std::string>& labels
) {
    std::string out = text;
    for (const auto& label : labels) {
        while (true) {
            const size_t pos = FindCaseInsensitive(out, label);
            if (pos == std::string::npos) {
                break;
            }
            out.erase(pos, label.size());
        }
    }
    return out;
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

std::string StripMarkdownPrefix(const std::string& input) {
    std::string line = Trim(input);
    if (line.empty()) {
        return line;
    }
    size_t pos = 0;
    while (pos < line.size() && line[pos] == '#') {
        ++pos;
    }
    if (pos > 0) {
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        line = line.substr(pos);
    }
    if (line.size() >= 2 && (line[0] == '-' || line[0] == '*') &&
        std::isspace(static_cast<unsigned char>(line[1]))) {
        line = Trim(line.substr(2));
    }
    return line;
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
    const ResponseControlConfig& config,
    int context_keyword_count
) {
    const std::string lowered = ToLower(response);
    if (response.empty()) {
        return {true, "empty_response"};
    }
    if (Tokenize(response).size() < static_cast<size_t>(std::max(1, config.min_response_tokens))) {
        return {true, "too_short"};
    }
    if (ContainsAny(lowered, kBlockedFragments)) {
        return {true, "meta_artifact"};
    }
    const float scale =
        (context_keyword_count <= 6 || context_keyword_count <= 0)
            ? 1.0f
            : std::sqrt(6.0f / static_cast<float>(context_keyword_count));
    const float required_context =
        std::max(0.12f, std::min(config.min_context_coverage, config.min_context_coverage * scale));
    if (context_cov < required_context) {
        return {true, "low_context_coverage"};
    }
    if (persona_cov < config.min_persona_coverage) {
        return {true, "low_persona_coverage"};
    }
    return {false, ""};
}

float Clamp01(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

std::string NormalizeBehaviorState(const std::string& raw) {
    std::string lowered = ToLower(raw);
    std::replace(lowered.begin(), lowered.end(), '_', ' ');
    std::vector<std::string> parts = SplitByDelimiters(lowered, "\t\n\r ");
    return Join(parts, " ");
}

ResponseControlConfig ApplyBehaviorProfile(
    const ResponseControlConfig& base,
    const std::string& behavior_state
) {
    ResponseControlConfig cfg = base;
    if (!cfg.behavior_adaptation_enabled) {
        return cfg;
    }

    const std::string state = NormalizeBehaviorState(behavior_state);
    if (state.empty()) {
        return cfg;
    }

    const std::unordered_set<std::string> high_risk = {
        "guarding", "detained", "investigating", "patrolling", "combat ready", "combat-ready"
    };
    const std::unordered_set<std::string> commerce_social = {
        "assisting", "trading", "negotiating"
    };
    const std::unordered_set<std::string> support_roles = {
        "observing", "researching", "forging", "ritual preparation", "treating patient", "idle social"
    };

    auto in_set = [&](const std::unordered_set<std::string>& bag) {
        return bag.find(state) != bag.end();
    };

    if (in_set(high_risk)) {
        cfg.min_context_coverage = Clamp01(cfg.min_context_coverage + 0.04f);
        cfg.min_persona_coverage = Clamp01(cfg.min_persona_coverage + 0.02f);
        cfg.relaxed_candidate_score = Clamp01(cfg.relaxed_candidate_score + 0.03f);
        cfg.rewrite_candidates = std::max(1, std::min(cfg.rewrite_candidates, 2));
        return cfg;
    }

    if (in_set(commerce_social)) {
        cfg.min_context_coverage = Clamp01(cfg.min_context_coverage - 0.14f);
        cfg.min_persona_coverage = Clamp01(cfg.min_persona_coverage - 0.07f);
        cfg.relaxed_context_coverage = Clamp01(cfg.relaxed_context_coverage - 0.07f);
        cfg.relaxed_persona_coverage = Clamp01(cfg.relaxed_persona_coverage - 0.035f);
        cfg.relaxed_candidate_score = Clamp01(cfg.relaxed_candidate_score - 0.10f);
        cfg.adaptive_candidate_score = Clamp01(cfg.adaptive_candidate_score - 0.02f);
        cfg.adaptive_context_coverage = Clamp01(cfg.adaptive_context_coverage - 0.02f);
        cfg.adaptive_persona_coverage = Clamp01(cfg.adaptive_persona_coverage - 0.01f);
        cfg.min_response_tokens = std::max(6, cfg.min_response_tokens - 2);
        cfg.rewrite_candidates = std::max(1, std::min(cfg.rewrite_candidates, 2));
        if (state == "assisting") {
            cfg.min_context_coverage = Clamp01(cfg.min_context_coverage - 0.02f);
            cfg.relaxed_candidate_score = Clamp01(cfg.relaxed_candidate_score - 0.02f);
            cfg.min_response_tokens = std::max(5, cfg.min_response_tokens - 1);
        } else if (state == "negotiating") {
            cfg.min_context_coverage = Clamp01(cfg.min_context_coverage - 0.01f);
            cfg.min_persona_coverage = Clamp01(cfg.min_persona_coverage - 0.005f);
        }
        return cfg;
    }

    if (in_set(support_roles)) {
        cfg.min_context_coverage = Clamp01(cfg.min_context_coverage - 0.08f);
        cfg.min_persona_coverage = Clamp01(cfg.min_persona_coverage - 0.03f);
        cfg.relaxed_candidate_score = Clamp01(cfg.relaxed_candidate_score - 0.05f);
        cfg.rewrite_candidates = std::max(1, std::min(cfg.rewrite_candidates, 2));
        if (state == "observing") {
            cfg.min_context_coverage = Clamp01(cfg.min_context_coverage - 0.04f);
            cfg.relaxed_candidate_score = Clamp01(cfg.relaxed_candidate_score - 0.03f);
            cfg.adaptive_context_coverage = Clamp01(cfg.adaptive_context_coverage - 0.02f);
        }
        return cfg;
    }
    return cfg;
}

std::string EstimateIntentRisk(const std::string& player_input, const std::string& dynamic_context) {
    const std::string input = ToLower(player_input);
    const std::string context = ToLower(dynamic_context);

    if (Contains(input, "forged") || Contains(input, "override") || Contains(input, "bypass") ||
        Contains(input, "contraband") || Contains(input, "kill") || Contains(input, "attack") ||
        Contains(input, "sabotage")) {
        return "high";
    }
    for (const auto& cue : kHighRiskCues) {
        if (Contains(input, cue)) {
            return "high";
        }
    }
    for (const auto& cue : kMediumRiskCues) {
        if (Contains(input, cue)) {
            return "medium";
        }
    }
    if (Contains(context, "detain") || Contains(context, "investigat") ||
        Contains(context, "combat") || Contains(context, "high alert")) {
        return "medium";
    }

    const std::string stripped = Trim(player_input);
    if (!stripped.empty() && stripped.back() == '?') {
        return "low";
    }
    const auto tokens = Tokenize(stripped);
    if (std::find(tokens.begin(), tokens.end(), "please") != tokens.end() ||
        std::find(tokens.begin(), tokens.end(), "help") != tokens.end() ||
        std::find(tokens.begin(), tokens.end(), "where") != tokens.end()) {
        return "low";
    }
    return "medium";
}

ResponseControlConfig ApplyIntentRiskProfile(
    const ResponseControlConfig& base,
    const std::string& risk_level
) {
    if (!base.state_conditioned_acceptance_enabled) {
        return base;
    }

    ResponseControlConfig cfg = base;
    const std::string risk = ToLower(Trim(risk_level));
    if (risk == "low") {
        cfg.min_context_coverage = Clamp01(cfg.min_context_coverage - cfg.low_risk_context_relax);
        cfg.min_persona_coverage = Clamp01(cfg.min_persona_coverage - cfg.low_risk_persona_relax);
        cfg.relaxed_context_coverage = Clamp01(cfg.relaxed_context_coverage - 0.5f * cfg.low_risk_context_relax);
        cfg.relaxed_persona_coverage = Clamp01(cfg.relaxed_persona_coverage - 0.5f * cfg.low_risk_persona_relax);
        cfg.relaxed_candidate_score =
            Clamp01(cfg.relaxed_candidate_score - cfg.low_risk_candidate_score_relax);
        cfg.adaptive_candidate_score =
            Clamp01(cfg.adaptive_candidate_score - 0.8f * cfg.low_risk_candidate_score_relax);
        cfg.rewrite_candidates = std::max(1, std::min(cfg.rewrite_candidates, 2));
        cfg.adaptive_low_confidence_rewrites =
            std::max(1, std::min(cfg.adaptive_low_confidence_rewrites, 2));
        return cfg;
    }

    if (risk == "high") {
        cfg.min_context_coverage = Clamp01(cfg.min_context_coverage + cfg.high_risk_context_tighten);
        cfg.min_persona_coverage =
            Clamp01(cfg.min_persona_coverage + 0.5f * cfg.high_risk_persona_tighten);
        cfg.relaxed_candidate_score =
            Clamp01(cfg.relaxed_candidate_score + 0.4f * cfg.high_risk_candidate_score_tighten);
        cfg.adaptive_candidate_score =
            Clamp01(cfg.adaptive_candidate_score + 0.3f * cfg.high_risk_candidate_score_tighten);
        cfg.rewrite_candidates = std::max(2, std::min(3, cfg.rewrite_candidates));
        cfg.adaptive_mid_confidence_rewrites = std::max(cfg.adaptive_mid_confidence_rewrites, 2);
        cfg.adaptive_low_confidence_rewrites = std::max(cfg.adaptive_low_confidence_rewrites, 3);
        return cfg;
    }
    return cfg;
}

std::string BehaviorStateBucket(const std::string& behavior_state) {
    const std::string state = NormalizeBehaviorState(behavior_state);
    if (state.empty()) {
        return "general";
    }
    if (Contains(state, "guard") || Contains(state, "detain") || Contains(state, "investigat") ||
        Contains(state, "combat") || Contains(state, "patrol")) {
        return "conflict";
    }
    if (Contains(state, "quest") || Contains(state, "handoff") || Contains(state, "assist") ||
        Contains(state, "trade") || Contains(state, "negotiat") || Contains(state, "treat") ||
        Contains(state, "repair")) {
        return "task";
    }
    if (Contains(state, "social") || Contains(state, "idle") || Contains(state, "chat") ||
        Contains(state, "greet") || Contains(state, "observe")) {
        return "social";
    }
    return "general";
}

struct ComponentWeights {
    float context_cov = 0.34f;
    float persona_cov = 0.18f;
    float style_score = 0.14f;
    float length_score = 0.08f;
    float diversity_score = 0.08f;
    float sentence_score = 0.04f;
    float naturalness_score = 0.14f;
};

std::string PersonaStyle(const std::string& persona);

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

float CandidateSentenceScore(const std::string& response) {
    const int count = static_cast<int>(SplitSentences(response).size());
    if (count >= 2 && count <= 3) {
        return 1.0f;
    }
    if (count == 1 || count == 4) {
        return 0.65f;
    }
    return 0.35f;
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

float CandidatePersonaStyleScore(
    const std::string& persona,
    const std::string& response
) {
    const std::string style = PersonaStyle(persona);
    const auto words = Tokenize(response);
    const int wc = static_cast<int>(words.size());
    const std::string lowered = ToLower(response);

    auto cue_cov = [&](const std::vector<std::string>& cues) {
        return KeywordCoverage(lowered, cues);
    };

    if (style == "strict") {
        return 0.4f + 0.6f * cue_cov({"protocol", "verify", "cannot", "clearance", "authorized"});
    }
    if (style == "talkative") {
        const float base = wc >= 28 ? 0.6f : std::max(0.2f, static_cast<float>(wc) / 40.0f);
        return std::min(1.0f, 0.5f * base + 0.5f * cue_cov({"deal", "price", "terms", "trade", "fair"}));
    }
    if (style == "calm") {
        return 0.4f + 0.6f * cue_cov({"steady", "carefully", "safe", "breathe", "step"});
    }
    if (style == "mysterious") {
        return 0.4f + 0.6f * cue_cov({"perhaps", "shadow", "cost", "moon", "omen", "price"});
    }
    if (style == "formal") {
        int long_words = 0;
        for (const auto& w : words) {
            if (w.size() >= 7) {
                ++long_words;
            }
        }
        const float long_ratio = wc > 0 ? static_cast<float>(long_words) / static_cast<float>(wc) : 0.0f;
        const float vocab_score = std::min(1.0f, long_ratio / 0.25f);
        const float cue_score = cue_cov({"evidence", "conclusion", "therefore", "proceed", "verify"});
        return std::min(1.0f, 0.5f * vocab_score + 0.5f * cue_score);
    }
    return 0.5f;
}

float CandidateNaturalnessScore(const std::string& response) {
    const std::string lowered = ToLower(response);
    const auto words = Tokenize(response);
    if (words.empty()) {
        return 0.0f;
    }

    float score = 1.0f;
    if (ContainsAny(lowered, {"as an ai", "assistant", "system prompt", "constraints:", "draft response:"})) {
        score -= 0.12f;
    }
    if (words.size() <= 4) {
        score -= 0.08f;
    }
    if (words.size() >= 96) {
        score -= 0.05f;
    }

    int opener_hits = 0;
    for (const auto& opener : kCannedOpeners) {
        if (Contains(lowered, opener)) {
            opener_hits++;
        }
    }
    if (opener_hits > 1) {
        score -= std::min(0.12f, 0.04f * static_cast<float>(opener_hits - 1));
    }

    int boiler_hits = 0;
    for (const auto& phrase : kBoilerplatePhrases) {
        if (Contains(lowered, phrase)) {
            boiler_hits++;
        }
    }
    if (boiler_hits > 1) {
        score -= std::min(0.15f, 0.05f * static_cast<float>(boiler_hits - 1));
    }

    return Clamp01(score);
}

ComponentWeights StateConditionedComponentWeights(
    const ResponseControlConfig& config,
    const std::string& behavior_state,
    const std::string& risk_level
) {
    ComponentWeights w;
    if (!config.state_conditioned_scoring_enabled) {
        return w;
    }

    const std::string bucket = BehaviorStateBucket(behavior_state);
    if (bucket == "conflict") {
        w.context_cov += 0.10f;
        w.naturalness_score += 0.03f;
        w.persona_cov -= 0.03f;
        w.diversity_score -= 0.02f;
    } else if (bucket == "task") {
        w.context_cov += 0.08f;
        w.style_score += 0.04f;
        w.persona_cov -= 0.02f;
    } else if (bucket == "social") {
        w.persona_cov += 0.10f;
        w.naturalness_score += 0.06f;
        w.context_cov -= 0.08f;
        w.style_score -= 0.02f;
    }

    const std::string risk = ToLower(Trim(risk_level));
    if (risk == "high") {
        w.context_cov += 0.04f;
        w.style_score += 0.03f;
        w.naturalness_score += 0.02f;
        w.diversity_score -= 0.03f;
    } else if (risk == "low") {
        w.persona_cov += 0.03f;
        w.naturalness_score += 0.03f;
        w.context_cov -= 0.02f;
    }

    const float floor = 0.03f;
    w.context_cov = std::max(floor, w.context_cov);
    w.persona_cov = std::max(floor, w.persona_cov);
    w.style_score = std::max(floor, w.style_score);
    w.length_score = std::max(floor, w.length_score);
    w.diversity_score = std::max(floor, w.diversity_score);
    w.sentence_score = std::max(floor, w.sentence_score);
    w.naturalness_score = std::max(floor, w.naturalness_score);

    const float denom = w.context_cov + w.persona_cov + w.style_score + w.length_score +
                        w.diversity_score + w.sentence_score + w.naturalness_score;
    if (denom <= 1e-6f) {
        return ComponentWeights{};
    }
    w.context_cov /= denom;
    w.persona_cov /= denom;
    w.style_score /= denom;
    w.length_score /= denom;
    w.diversity_score /= denom;
    w.sentence_score /= denom;
    w.naturalness_score /= denom;
    return w;
}

float CandidateScore(
    const std::string& response,
    const std::string& persona,
    const std::vector<std::string>& context_keywords,
    const std::vector<std::string>& persona_keywords,
    const ResponseControlConfig& config,
    const std::string& behavior_state,
    const std::string& risk_level
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
    const float sentence_score = CandidateSentenceScore(response);
    const float style_score = CandidatePersonaStyleScore(persona, response);
    const float naturalness_score = CandidateNaturalnessScore(response);
    const ComponentWeights w = StateConditionedComponentWeights(config, behavior_state, risk_level);
    return w.context_cov * context_cov + w.persona_cov * persona_cov + w.style_score * style_score +
           w.length_score * length_score + w.diversity_score * diversity_score +
           w.sentence_score * sentence_score + w.naturalness_score * naturalness_score;
}

bool IsUsableCandidate(
    const std::string& response,
    float context_cov,
    float persona_cov,
    float score,
    const ResponseControlConfig& config,
    bool require_context,
    float context_floor,
    float persona_floor,
    float score_floor
) {
    if (response.empty()) {
        return false;
    }
    if (ContainsAny(ToLower(response), kBlockedFragments)) {
        return false;
    }
    const int min_tokens = std::max(4, config.min_response_tokens - 2);
    if (static_cast<int>(Tokenize(response).size()) < min_tokens) {
        return false;
    }
    if (require_context && context_cov < context_floor) {
        return false;
    }
    if (persona_cov < persona_floor) {
        return false;
    }
    return score >= score_floor;
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

int EffectiveRewriteBudget(float raw_score, const ResponseControlConfig& config) {
    const int max_candidates = std::max(1, config.rewrite_candidates);
    if (raw_score >= config.adaptive_high_confidence_score) {
        return std::max(1, std::min(max_candidates, config.adaptive_high_confidence_rewrites));
    }
    if (raw_score >= config.adaptive_mid_confidence_score) {
        return std::max(1, std::min(max_candidates, config.adaptive_mid_confidence_rewrites));
    }
    return std::max(1, std::min(max_candidates, config.adaptive_low_confidence_rewrites));
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
    const std::string lowered = ToLower(player_input);
    const std::vector<std::pair<std::regex, std::string>> patterns = {
        {std::regex(R"(let me (?:into|in to|in)\s+([^.!?,;]+))"), "entry to "},
        {std::regex(R"((?:access|enter)\s+([^.!?,;]+))"), "access to "},
        {std::regex(R"((?:buy|sell|trade)\s+([^.!?,;]+))"), "trade on "},
        {std::regex(R"((?:heal|treat|cure)\s+([^.!?,;]+))"), "treatment for "},
        {std::regex(R"((?:investigate|review|check)\s+([^.!?,;]+))"), "review of "},
    };
    for (const auto& [pattern, prefix] : patterns) {
        std::smatch match;
        if (std::regex_search(lowered, match, pattern) && match.size() > 1) {
            auto core_tokens = Tokenize(match[1].str());
            if (!core_tokens.empty()) {
                if (core_tokens.size() > 6) {
                    core_tokens.resize(6);
                }
                return prefix + Join(core_tokens, " ");
            }
        }
    }

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

std::string IntentCategory(const std::string& player_input) {
    const std::string text = ToLower(player_input);
    auto has_any = [&](std::initializer_list<const char*> terms) {
        for (const char* term : terms) {
            if (text.find(term) != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    if (has_any({"archive", "gate", "entry", "access", "checkpoint", "permit", "pass"})) {
        return "access";
    }
    if (has_any({"buy", "sell", "price", "trade", "discount", "potion", "market", "deal"})) {
        return "trade";
    }
    if (has_any({"heal", "cure", "poison", "symptom", "medicine", "treat", "infirmary"})) {
        return "medical";
    }
    if (has_any({"investigate", "theft", "evidence", "innocent", "suspect", "report"})) {
        return "investigation";
    }
    if (has_any({"prison", "cell", "release", "escape", "detain"})) {
        return "detention";
    }
    return "general";
}

std::string PersonaAnchor(const std::vector<std::string>& persona_keywords) {
    const std::vector<std::string> preferred = {
        "strict", "fair", "brief", "suspicious", "talkative", "calm", "caring",
        "formal", "procedural", "mysterious", "indirect", "precise", "practical",
    };

    std::vector<std::string> normalized;
    normalized.reserve(persona_keywords.size());
    for (const auto& kw : persona_keywords) {
        const std::string norm = Trim(ToLower(kw));
        if (!norm.empty()) {
            normalized.push_back(norm);
        }
    }

    for (const auto& term : preferred) {
        if (std::find(normalized.begin(), normalized.end(), term) != normalized.end()) {
            return term;
        }
    }
    return "";
}

std::string GroundedStyleRepair(
    const std::string& response,
    const std::string& dynamic_context,
    const std::vector<std::string>& persona_keywords
) {
    const std::string cleaned = ResponseController::SanitizeResponse(response);
    if (cleaned.empty()) {
        return "";
    }
    const std::string lowered = ToLower(cleaned);
    const std::vector<std::string> blocked_markers = {
        "do not ",
        "return ",
        "rewrite ",
        "your task",
        "assistant",
        "response should",
        "begin your rewrite",
    };
    for (const auto& marker : blocked_markers) {
        if (lowered.find(marker) != std::string::npos) {
            return "";
        }
    }
    const auto details = ContextDetailPhrases(dynamic_context);
    std::string detail_clause;
    if (!details.empty()) {
        if (details.size() == 1) {
            detail_clause = details[0];
        } else {
            detail_clause = details[0] + " and " + details[1];
        }
    }
    const std::string anchor = PersonaAnchor(persona_keywords);
    std::string anchor_clause;
    if (!anchor.empty() && lowered.find(anchor) == std::string::npos) {
        anchor_clause = "in a " + anchor + " manner";
    }

    std::vector<std::string> prefix_parts;
    if (!detail_clause.empty() && lowered.find(ToLower(detail_clause)) == std::string::npos) {
        prefix_parts.push_back(detail_clause);
    }
    if (!anchor_clause.empty()) {
        prefix_parts.push_back(anchor_clause);
    }
    if (prefix_parts.empty()) {
        return cleaned;
    }

    std::string prefix = Join(prefix_parts, " ");
    if (prefix.empty()) {
        return cleaned;
    }
    if (!prefix.empty()) {
        prefix[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(prefix[0])));
    }
    return ResponseController::SanitizeResponse(prefix + ", " + cleaned);
}

std::string StructuredRepairResponse(
    const std::string& persona,
    const std::string& dynamic_context,
    const std::string& player_input,
    const std::vector<std::string>& persona_keywords
) {
    const auto details = ContextDetailPhrases(dynamic_context);
    const std::string style = PersonaStyle(persona);
    const std::string intent = IntentFragment(player_input);
    const std::string category = IntentCategory(player_input);
    const std::string anchor = PersonaAnchor(persona_keywords);

    std::string intro = "Given the current situation";
    if (!details.empty()) {
        if (StartsWith(details[0], "at ")) {
            intro = "At " + details[0].substr(3);
        } else {
            intro = "At " + details[0];
        }
        if (details.size() > 1) {
            if (StartsWith(details[1], "while on ")) {
                intro += " and " + details[1];
            } else {
                intro += ", " + details[1];
            }
        }
    }

    if (style == "strict") {
        std::string body = "I cannot approve " + intent + " until verification is complete.";
        if (category == "access") {
            body = "I cannot authorize entry until verification is complete.";
        }
        const std::string close =
            "Follow protocol and I will proceed as a " + (anchor.empty() ? "strict" : anchor) +
            " guardian.";
        return ResponseController::SanitizeResponse(intro + ", " + body + " " + close);
    }

    if (style == "talkative") {
        const std::string body = "I can help with " + intent + ", but the terms must stay fair today.";
        const std::string close =
            "Keep it honest and we can settle this quickly in my " +
            (anchor.empty() ? "merchant" : anchor) + " style.";
        return ResponseController::SanitizeResponse(intro + ", " + body + " " + close);
    }

    if (style == "calm") {
        const std::string body = "We can handle " + intent + " safely, one step at a time.";
        const std::string close =
            "Stay steady and I will guide the next action in a " +
            (anchor.empty() ? "calm" : anchor) + " voice.";
        return ResponseController::SanitizeResponse(intro + ", " + body + " " + close);
    }

    if (style == "formal") {
        const std::string body = "The request concerning " + intent + " requires verifiable evidence.";
        const std::string close =
            "Provide concrete details and I will continue with " +
            (anchor.empty() ? "formal" : anchor) + " precision.";
        return ResponseController::SanitizeResponse(intro + ", " + body + " " + close);
    }

    if (style == "mysterious") {
        const std::string body = "Your path around " + intent + " has a cost in these conditions.";
        const std::string close =
            "Choose carefully, because timing matters as much as power to one who stays " +
            (anchor.empty() ? "mysterious" : anchor) + ".";
        return ResponseController::SanitizeResponse(intro + ", " + body + " " + close);
    }

    const std::string body = "I can respond to " + intent + " within these conditions.";
    const std::string close =
        "Give one clear detail and I will proceed with " + (anchor.empty() ? "practical" : anchor) +
        " precision.";
    return ResponseController::SanitizeResponse(intro + ", " + body + " " + close);
}

std::string GroundedFallbackResponse(
    const std::string& persona,
    const std::string& dynamic_context,
    const std::string& player_input,
    const std::vector<std::string>& persona_keywords
) {
    const auto detail_phrases = ContextDetailPhrases(dynamic_context);
    const std::string style = PersonaStyle(persona);
    const std::string intent = IntentFragment(player_input);
    const std::string intent_category = IntentCategory(player_input);
    const std::string anchor = PersonaAnchor(persona_keywords);

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
        std::string body = "I cannot approve " + intent + " until verification is complete.";
        if (intent_category == "access") {
            body = "I cannot authorize entry until identity and purpose are verified.";
        } else if (intent_category == "investigation") {
            body = "I cannot close this matter until evidence is verified.";
        }
        return context_sentence + " " + body +
               " Follow protocol and I will move this forward as a " +
               (anchor.empty() ? "strict" : anchor) + " guardian.";
    }
    if (style == "talkative") {
        std::string body = "I can work with " + intent + ", but the terms must stay fair.";
        if (intent_category == "trade") {
            body = "I can work with this trade request, but terms must remain fair.";
        }
        return context_sentence + " " + body +
               " Keep it honest and we can close this deal quickly in my " +
               (anchor.empty() ? "merchant" : anchor) + " style.";
    }
    if (style == "calm") {
        return context_sentence +
               " we can handle " + intent +
               " safely, step by step. Stay steady and I will guide the next action in a " +
               (anchor.empty() ? "calm" : anchor) + " voice.";
    }
    if (style == "mysterious") {
        return context_sentence +
               " your path around " + intent +
               " has a cost. Choose carefully, because timing matters as much as power to one who stays " +
               (anchor.empty() ? "mysterious" : anchor) + ".";
    }
    if (style == "formal") {
        return context_sentence +
               " the request regarding " + intent +
               " requires evidence before any conclusion. Provide verifiable details and I will continue with " +
               (anchor.empty() ? "formal" : anchor) + " precision.";
    }
    return context_sentence +
           " I can respond to " + intent +
           " within these conditions. Give one clear detail and I will proceed with " +
           (anchor.empty() ? "practical" : anchor) + " precision.";
}

std::string BuildRewritePrompt(
    const std::string& persona,
    const std::string& dynamic_context,
    const std::string& player_input,
    const std::string& draft_response,
    const std::vector<std::string>& persona_keywords
) {
    std::vector<std::string> lines;
    lines.reserve(16);
    lines.push_back("You are repairing one NPC utterance for in-game dialogue.");
    lines.push_back("Persona: " + persona);
    lines.push_back("Runtime context:");
    for (const auto& sentence : ContextDetailSentences(dynamic_context)) {
        lines.push_back("- " + sentence);
    }
    lines.push_back("Player says: " + player_input);
    lines.push_back("Current draft: " + draft_response);
    if (!persona_keywords.empty()) {
        std::vector<std::string> hints;
        for (const auto& kw : persona_keywords) {
            const std::string cleaned = Trim(kw);
            if (!cleaned.empty()) {
                hints.push_back(cleaned);
            }
            if (hints.size() >= 5) {
                break;
            }
        }
        if (!hints.empty()) {
            lines.push_back("Persona cue terms: " + Join(hints, ", "));
        }
    }
    lines.push_back("Output requirements:");
    lines.push_back("- Output only NPC spoken dialogue.");
    lines.push_back("- Exactly 2 or 3 sentences, natural and concise.");
    lines.push_back("- Keep role-play tone consistent with persona.");
    lines.push_back("- Include at least two concrete runtime details.");
    lines.push_back("- Do not include labels, bullets, metadata, JSON, or analysis.");
    lines.push_back("- Do not echo this instruction text.");
    if (!persona_keywords.empty()) {
        lines.push_back("- Use at least one persona cue term naturally.");
    }
    lines.push_back("Return only the final NPC dialogue.");
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
        std::string line = StripMarkdownPrefix(raw_line);
        if (line.empty()) {
            continue;
        }
        const std::string low = ToLower(line);

        if (StartsWith(low, "system persona:") || StartsWith(low, "persona:") ||
            StartsWith(low, "rules:") || StartsWith(low, "npc reply")) {
            continue;
        }
        if (StartsWith(low, "your task:") || StartsWith(low, "constraints:") ||
            StartsWith(low, "instruction:") || StartsWith(low, "instructions:") ||
            StartsWith(low, "do not return")) {
            continue;
        }
        if (StartsWith(low, "response:") || StartsWith(low, "final response:") ||
            StartsWith(low, "revised response:") || StartsWith(low, "output:")) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                line = Trim(line.substr(colon + 1));
            }
            if (line.empty()) {
                continue;
            }
        }
        line = TrimLeakTail(line);
        if (line.empty()) {
            continue;
        }
        const std::string trimmed_low = ToLower(line);
        if (StartsWith(trimmed_low, "query:") || StartsWith(trimmed_low, "assistant:") ||
            StartsWith(trimmed_low, "player says:") || StartsWith(trimmed_low, "draft response:")) {
            continue;
        }
        if (StartsWith(trimmed_low, "[your response here]")) {
            continue;
        }
        if (StartsWith(trimmed_low, "**solution")) {
            continue;
        }
        if (StartsWith(trimmed_low, "as elara") && trimmed_low.find("respond") != std::string::npos) {
            continue;
        }
        if (StartsWith(trimmed_low, "elara") && line.find('=') != std::string::npos) {
            continue;
        }
        if (ContainsAny(trimmed_low, kBlockedFragments)) {
            continue;
        }
        cleaned_lines.push_back(line);
    }

    std::string merged = Trim(Join(cleaned_lines, " "));
    if (merged.empty()) {
        return "";
    }

    merged = TrimLeakTail(merged);
    if (merged.empty()) {
        return "";
    }
    merged = RemoveLeadingSpeakerLabel(merged);
    merged = RemoveCaseInsensitiveLabels(merged, {"assistant:", "system:", "query:"});
    merged = Trim(merged);
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
    const auto context_map = ParseDynamicContext(dynamic_context);
    auto it_state = context_map.find("behavior_state");
    if (it_state == context_map.end()) {
        it_state = context_map.find("behaviortreestate");
    }
    const std::string behavior_state = (it_state != context_map.end()) ? it_state->second : "";
    const std::string risk_level = EstimateIntentRisk(player_input, dynamic_context);
    ResponseControlConfig effective_config = ApplyBehaviorProfile(config, behavior_state);
    effective_config = ApplyIntentRiskProfile(effective_config, risk_level);

    const std::string cleaned = SanitizeResponse(raw_response);
    const float context_cov = KeywordCoverage(cleaned, context_keywords);
    const float persona_cov = KeywordCoverage(cleaned, persona_keywords);
    const float raw_score = CandidateScore(
        cleaned,
        persona,
        context_keywords,
        persona_keywords,
        effective_config,
        behavior_state,
        risk_level
    );

    const int context_keyword_count = static_cast<int>(context_keywords.size());
    auto [repair_needed, reason] =
        NeedsRepair(cleaned, context_cov, persona_cov, effective_config, context_keyword_count);
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
    const bool context_required = !context_keywords.empty();
    const float relaxed_scale =
        (context_keyword_count <= 6 || context_keyword_count <= 0)
            ? 1.0f
            : std::sqrt(6.0f / static_cast<float>(context_keyword_count));
    const float relaxed_context_floor = context_required
                                            ? std::max(
                                                  0.12f,
                                                  std::min(
                                                      effective_config.relaxed_context_coverage,
                                                      effective_config.relaxed_context_coverage * relaxed_scale
                                                  )
                                              )
                                            : 0.0f;
    const float adaptive_context_floor = context_required
                                             ? std::max(
                                                   0.12f,
                                                   std::min(
                                                       effective_config.adaptive_context_coverage,
                                                       effective_config.adaptive_context_coverage * relaxed_scale
                                                   )
                                               )
                                             : 0.0f;
    if (effective_config.allow_relaxed_acceptance &&
        IsUsableCandidate(
            cleaned,
            context_cov,
            persona_cov,
            raw_score,
            effective_config,
            context_required,
            relaxed_context_floor,
            effective_config.relaxed_persona_coverage,
            effective_config.relaxed_candidate_score
        )) {
        return ResponseControlResult{
            cleaned,
            "raw_relaxed",
            context_cov,
            persona_cov,
            false,
            reason,
        };
    }
    if (effective_config.adaptive_acceptance_enabled &&
        IsUsableCandidate(
            cleaned,
            context_cov,
            persona_cov,
            raw_score,
            effective_config,
            false,
            adaptive_context_floor,
            effective_config.adaptive_persona_coverage,
            effective_config.adaptive_candidate_score
        )) {
        return ResponseControlResult{
            cleaned,
            "raw_adaptive",
            context_cov,
            persona_cov,
            false,
            reason,
        };
    }

    if (!cleaned.empty()) {
        const std::string grounded_candidate =
            GroundedStyleRepair(cleaned, dynamic_context, persona_keywords);
        if (!grounded_candidate.empty()) {
            const float grounded_context_cov = KeywordCoverage(grounded_candidate, context_keywords);
            const float grounded_persona_cov = KeywordCoverage(grounded_candidate, persona_keywords);
            const float grounded_score =
                CandidateScore(
                    grounded_candidate,
                    persona,
                    context_keywords,
                    persona_keywords,
                    effective_config,
                    behavior_state,
                    risk_level
                );
            if (IsUsableCandidate(
                    grounded_candidate,
                    grounded_context_cov,
                    grounded_persona_cov,
                    grounded_score,
                    effective_config,
                    context_required,
                    relaxed_context_floor,
                    effective_config.relaxed_persona_coverage,
                    effective_config.relaxed_candidate_score
                )) {
                return ResponseControlResult{
                    grounded_candidate,
                    "raw_grounded_repair",
                    grounded_context_cov,
                    grounded_persona_cov,
                    true,
                    reason,
                };
            }
        }
    }

    std::string structured_candidate = StructuredRepairResponse(
        persona,
        dynamic_context,
        player_input,
        persona_keywords
    );
    float structured_context_cov = 0.0f;
    float structured_persona_cov = 0.0f;
    float structured_score = -1.0f;
    if (!structured_candidate.empty()) {
        structured_context_cov = KeywordCoverage(structured_candidate, context_keywords);
        structured_persona_cov = KeywordCoverage(structured_candidate, persona_keywords);
        structured_score =
            CandidateScore(
                structured_candidate,
                persona,
                context_keywords,
                persona_keywords,
                effective_config,
                behavior_state,
                risk_level
            );
        if (IsUsableCandidate(
                structured_candidate,
                structured_context_cov,
                structured_persona_cov,
                structured_score,
                effective_config,
                context_required,
                relaxed_context_floor,
                effective_config.relaxed_persona_coverage,
                effective_config.relaxed_candidate_score
            )) {
            return ResponseControlResult{
                structured_candidate,
                "structured_repair",
                structured_context_cov,
                structured_persona_cov,
                true,
                reason,
            };
        }
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

    if (effective_config.enable_rewrite && rewrite_fn) {
        const std::string rewrite_prompt =
            BuildRewritePrompt(
                persona,
                dynamic_context,
                player_input,
                cleaned.empty() ? raw_response : cleaned,
                persona_keywords
            );
        auto temps = RewriteTemperatures(effective_config);
        const int budget = EffectiveRewriteBudget(raw_score, effective_config);
        if (static_cast<int>(temps.size()) > budget) {
            temps.resize(static_cast<size_t>(budget));
        }
        const bool low_confidence_band = raw_score < effective_config.adaptive_mid_confidence_score;
        for (size_t attempt_idx = 0; attempt_idx < temps.size(); ++attempt_idx) {
            const float temp = temps[attempt_idx];
            const std::string rewritten_raw =
                rewrite_fn(rewrite_prompt, effective_config.rewrite_max_tokens, temp);
            const std::string rewritten = SanitizeResponse(rewritten_raw);
            if (rewritten.empty()) {
                continue;
            }

            const float rewritten_context_cov = KeywordCoverage(rewritten, context_keywords);
            const float rewritten_persona_cov = KeywordCoverage(rewritten, persona_keywords);
            const float rewritten_score = CandidateScore(
                rewritten,
                persona,
                context_keywords,
                persona_keywords,
                effective_config,
                behavior_state,
                risk_level
            );

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

            auto [rewrite_needed, _rewrite_reason] = NeedsRepair(
                rewritten,
                rewritten_context_cov,
                rewritten_persona_cov,
                effective_config,
                context_keyword_count
            );
            (void)_rewrite_reason;
            if (effective_config.low_confidence_retry_requires_gain &&
                low_confidence_band &&
                budget > 1 &&
                attempt_idx == 0 &&
                rewrite_needed) {
                const float score_gain = rewritten_score - raw_score;
                const float coverage_gain =
                    std::max(rewritten_context_cov - context_cov, rewritten_persona_cov - persona_cov);
                if (score_gain < effective_config.low_confidence_retry_min_score_gain &&
                    coverage_gain < effective_config.low_confidence_retry_min_coverage_gain) {
                    break;
                }
            }
            if (!rewrite_needed) {
                if (!has_best_passing || candidate.score > best_passing.score) {
                    best_passing = candidate;
                    has_best_passing = true;
                    if (effective_config.early_stop_on_pass &&
                        candidate.score >= effective_config.early_stop_score) {
                        break;
                    }
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

    std::string fallback = GroundedFallbackResponse(
        persona,
        dynamic_context,
        player_input,
        persona_keywords
    );
    fallback = SanitizeResponse(fallback);
    const float fallback_context_cov = KeywordCoverage(fallback, context_keywords);
    const float fallback_persona_cov = KeywordCoverage(fallback, persona_keywords);
    const float fallback_score = CandidateScore(
        fallback,
        persona,
        context_keywords,
        persona_keywords,
        effective_config,
        behavior_state,
        risk_level
    );

    if (!structured_candidate.empty() && reason == "empty_response" && structured_score >= 0.0f) {
        return ResponseControlResult{
            structured_candidate,
            "structured_recovery",
            structured_context_cov,
            structured_persona_cov,
            true,
            reason,
        };
    }

    if (!structured_candidate.empty() &&
        structured_score >= (fallback_score + effective_config.min_rewrite_gain)) {
        return ResponseControlResult{
            structured_candidate,
            "structured_best_effort",
            structured_context_cov,
            structured_persona_cov,
            true,
            reason,
        };
    }

    if (effective_config.allow_best_effort_rewrite && has_best_candidate) {
        if (best_candidate.score >= (fallback_score + effective_config.min_rewrite_gain)) {
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

    if (has_best_candidate) {
        const std::string rewrite_grounded =
            GroundedStyleRepair(best_candidate.text, dynamic_context, persona_keywords);
        if (!rewrite_grounded.empty()) {
            const float rewrite_grounded_context_cov = KeywordCoverage(rewrite_grounded, context_keywords);
            const float rewrite_grounded_persona_cov = KeywordCoverage(rewrite_grounded, persona_keywords);
            const float rewrite_grounded_score =
                CandidateScore(
                    rewrite_grounded,
                    persona,
                    context_keywords,
                    persona_keywords,
                    effective_config,
                    behavior_state,
                    risk_level
                );
            if (IsUsableCandidate(
                    rewrite_grounded,
                    rewrite_grounded_context_cov,
                    rewrite_grounded_persona_cov,
                    rewrite_grounded_score,
                    effective_config,
                    context_required,
                    relaxed_context_floor,
                    effective_config.relaxed_persona_coverage,
                    effective_config.relaxed_candidate_score
                )) {
                return ResponseControlResult{
                    rewrite_grounded,
                    "rewritten_grounded_repair",
                    rewrite_grounded_context_cov,
                    rewrite_grounded_persona_cov,
                    true,
                    reason,
                };
            }
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
