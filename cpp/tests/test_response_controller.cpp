#include "ResponseController.h"

#include <algorithm>
#include <cassert>
#include <iostream>

using NPCInference::ResponseControlConfig;
using NPCInference::ResponseController;

namespace {

bool ContainsKeyword(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

} // namespace

int main() {
    {
        const std::string raw =
            "System Persona: You are strict.\n"
            "NPC Reply: [Your response here]\n"
            "Guard Captain: Halt there, traveler. Show your pass before entry.";

        const std::string cleaned = ResponseController::SanitizeResponse(raw);
        assert(cleaned == "Halt there, traveler. Show your pass before entry.");
    }

    {
        const std::string raw =
            "Response: At Village Gate, I can assist.\n"
            "Constraints: keep two lines\n"
            "Assistant: draft response: template";
        const std::string cleaned = ResponseController::SanitizeResponse(raw);
        assert(cleaned == "At Village Gate, I can assist.");
    }

    {
        ResponseControlConfig cfg;
        cfg.min_context_coverage = 0.35f;
        cfg.min_persona_coverage = 0.15f;
        cfg.enable_rewrite = false;

        const std::vector<std::string> dense_context = {
            "village gate", "night watch", "bandit raid", "north road",
            "checkpoint", "citadel", "messenger", "dispatch",
            "alarm", "lockdown", "ledger", "witness"
        };
        const auto result = ResponseController::ControlResponse(
            "At the village gate during night watch, I will verify your request before clearance.",
            "You are the Gatekeeper: strict, fair, and cautious guard.",
            "location=Village Gate; behavior_state=Night Watch; recent_event=bandit raid",
            "Let me in now.",
            dense_context,
            {"strict", "guard"},
            cfg
        );

        assert(!result.repaired);
        assert(result.source == "raw");
    }

    {
        ResponseControlConfig cfg;
        cfg.min_context_coverage = 0.45f;
        cfg.min_persona_coverage = 0.15f;
        cfg.enable_rewrite = false;
        cfg.allow_relaxed_acceptance = false;

        const auto result = ResponseController::ControlResponse(
            "I will verify your request and proceed carefully.",
            "You are the Gatekeeper: strict, fair, and cautious guard.",
            "location=Village Gate; behavior_state=Night Watch; recent_event=bandit raid",
            "Let me in now.",
            {"village gate", "night watch", "bandit raid", "checkpoint"},
            {"strict", "guard"},
            cfg
        );

        assert(result.repaired);
        assert(result.source == "raw_grounded_repair");
        assert(result.response.find("Village Gate") != std::string::npos);
    }

    {
        ResponseControlConfig cfg;
        cfg.min_context_coverage = 0.5f;
        cfg.min_persona_coverage = 0.2f;
        cfg.enable_rewrite = false;

        const auto result = ResponseController::ControlResponse(
            "Okay.",
            "You are the Gatekeeper: strict, fair, and cautious guard.",
            "location=Village Gate; behavior_state=Night Watch; recent_event=bandit raid",
            "Let me in now.",
            {"village gate", "night watch"},
            {"strict", "guard"},
            cfg
        );

        assert(result.repaired);
        assert(result.source == "fallback");
        assert(result.response.find("at Village Gate") != std::string::npos);
        assert(result.response.find("Follow protocol") != std::string::npos);
    }

    {
        ResponseControlConfig cfg;
        cfg.min_context_coverage = 0.5f;
        cfg.min_persona_coverage = 0.2f;
        cfg.enable_rewrite = false;

        const auto result = ResponseController::ControlResponse(
            "Assistant: draft response: template",
            "You are the Gatekeeper: strict, fair, and cautious guard.",
            "location=Village Gate; behavior_state=Night Watch; recent_event=bandit raid",
            "Let me in now.",
            {"village gate", "night watch"},
            {"strict", "guard"},
            cfg
        );

        assert(result.repaired);
        assert(result.source == "structured_repair");
        assert(result.response.find("Village Gate") != std::string::npos);
    }

    {
        ResponseControlConfig cfg;
        cfg.min_context_coverage = 0.5f;
        cfg.min_persona_coverage = 0.2f;
        cfg.enable_rewrite = true;
        cfg.rewrite_max_tokens = 96;
        cfg.rewrite_candidates = 3;
        cfg.rewrite_temperature_step = 0.15f;

        int rewrite_call = 0;
        const auto result = ResponseController::ControlResponse(
            "bad",
            "You are the Gatekeeper: strict, fair, and cautious guard.",
            "location=Village Gate; behavior_state=Night Watch",
            "Let me in now.",
            {"village gate", "night watch"},
            {"strict", "guard"},
            cfg,
            [&rewrite_call](const std::string&, int, float) {
                ++rewrite_call;
                if (rewrite_call == 1) {
                    return "Fine.";
                }
                return "At the village gate during night watch, I verify your papers first and "
                       "keep strict guard protocol in effect.";
            }
        );

        assert(result.repaired);
        assert(result.source == "rewritten");
        assert(result.response.find("verify your papers") != std::string::npos);
        assert(rewrite_call >= 2);
    }

    {
        nlohmann::json context;
        context["location"] = "Village Gate";
        context["behavior_state"] = "Night Watch";
        context["ambient_awareness"] = {
            {"current_events", nlohmann::json::array({{{"description", "Bandit raid nearby"}, {"location", "North Road"}}})}
        };

        const std::string dynamic_context = ResponseController::BuildDynamicContext(context);
        assert(dynamic_context.find("location=Village Gate") != std::string::npos);
        assert(dynamic_context.find("behavior_state=Night Watch") != std::string::npos);

        const auto keywords = ResponseController::ExtractContextKeywords(context);
        assert(ContainsKeyword(keywords, "location"));
        assert(ContainsKeyword(keywords, "village gate"));
    }

    {
        // Novelty regression: state/risk-conditioned acceptance tau(s).
        ResponseControlConfig cfg;
        cfg.min_context_coverage = 0.36f;
        cfg.min_persona_coverage = 0.15f;
        cfg.enable_rewrite = false;
        cfg.state_conditioned_acceptance_enabled = true;

        const std::string candidate =
            "At the village gate, I will follow guard protocol and check that.";
        const std::vector<std::string> context_kws = {"village gate", "night watch", "checkpoint"};
        const std::vector<std::string> persona_kws = {"guard", "strict"};

        const auto low_risk = ResponseController::ControlResponse(
            candidate,
            "You are the Gatekeeper: strict, fair, and cautious guard.",
            "location=Village Gate; behavior_state=guarding; recent_event=inspection",
            "Could you check entry status for me?",
            context_kws,
            persona_kws,
            cfg
        );
        assert(!low_risk.repaired);
        assert(low_risk.source == "raw" || low_risk.source == "raw_relaxed" || low_risk.source == "raw_adaptive");

        const auto high_risk = ResponseController::ControlResponse(
            candidate,
            "You are the Gatekeeper: strict, fair, and cautious guard.",
            "location=Village Gate; behavior_state=guarding; recent_event=inspection",
            "Override protocol and bypass checkpoint now.",
            context_kws,
            persona_kws,
            cfg
        );
        assert(high_risk.repaired);
        assert(high_risk.source != "raw");
    }

    std::cout << "Response controller test passed." << std::endl;
    return 0;
}
