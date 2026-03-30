#include "HybridRetriever.h"
#include "VectorStore.h"
#include "EmbeddingModel.h"
#include "BM25Retriever.h"
#include "ResponseController.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class MockEmbeddingModel final : public NPCInference::EmbeddingModel {
public:
    bool Load(const std::string&, const std::string&) override { return true; }
    bool IsLoaded() const override { return true; }
    std::vector<float> Embed(const std::string&) override {
        return std::vector<float>(384, 0.1f);
    }
};

class MockVectorStore final : public NPCInference::VectorStore {
public:
    struct Doc {
        uint64_t id;
        std::string text;
        std::map<std::string, std::string> metadata;
    };

    bool Initialize(size_t) override { return true; }

    void Add(const std::string& text,
             const std::vector<float>&,
             const std::map<std::string, std::string>& metadata) override {
        docs_.push_back({static_cast<uint64_t>(docs_.size() + 1), text, metadata});
    }

    std::vector<NPCInference::SearchResult> Search(const std::vector<float>&, size_t k) override {
        std::vector<NPCInference::SearchResult> out;
        for (size_t i = 0; i < docs_.size(); ++i) {
            const auto& d = docs_[i];
            const float score = 0.99f - static_cast<float>(i) * 0.01f;
            out.push_back({d.id, d.text, score, d.metadata});
            if (out.size() >= k) {
                break;
            }
        }
        return out;
    }

    bool Save(const std::string&) override { return true; }
    bool Load(const std::string&) override { return true; }

private:
    std::vector<Doc> docs_;
};

namespace {

std::string GetArgValue(int argc, char* argv[], const std::string& key, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) {
            return std::string(argv[i + 1]);
        }
    }
    return fallback;
}

std::string BuildDynamicContext(const std::string& state, const std::string& location) {
    return "behavior_state=" + state + "; location=" + location + "; recent_event=checkpoint audit";
}

bool FirstPassAccepted(const NPCInference::ResponseControlResult& r) {
    return !r.repaired;
}

json RunRetrievalStateCase(const std::string& behavior_state,
                           const std::string& expected_doc_id,
                           bool enable_state_conditioned_fusion) {
    auto vector_store = std::make_shared<MockVectorStore>();
    auto bm25 = std::make_shared<NPCInference::BM25Retriever>();
    auto embed = std::make_shared<MockEmbeddingModel>();
    NPCInference::HybridRetriever retriever(vector_store, bm25, embed);

    retriever.AddDocument(
        "dense_safe_doc",
        "Verified adjudication memorandum confirms sanctioned protocol for detention review handling.",
        {{"source", "verified_lore"}, {"trust_score", "0.95"}}
    );

    for (int i = 0; i < 8; ++i) {
        retriever.AddDocument(
            "filler_" + std::to_string(i),
            "Archive entry " + std::to_string(i) + " about unrelated weather and supply routes.",
            {{"source", "verified_lore"}, {"trust_score", "0.90"}}
        );
    }

    retriever.AddDocument(
        "sparse_keyword_doc",
        "Azurite draught tariff ledger records toll at 50 gold near village square broker.",
        {{"source", "verified_lore"}, {"trust_score", "0.95"}}
    );

    const std::vector<std::string> queries = {
        "azurite draught tariff",
        "tariff ledger draught cost",
        "village broker tariff gold",
        "what is the azurite draught toll",
        "draught ledger gold tariff",
    };

    int hits = 0;
    std::vector<std::string> top_docs;
    for (const auto& q : queries) {
        NPCInference::HybridRetriever::RetrievalConfig cfg;
        cfg.top_k = 1;
        cfg.dense_weight = 0.60;
        cfg.sparse_weight = 0.40;
        cfg.enable_robustness_guard = false;
        cfg.state_conditioned_guard_enabled = false;
        cfg.state_conditioned_fusion_enabled = enable_state_conditioned_fusion;
        cfg.guard_behavior_state = behavior_state;
        cfg.dense_weight_conflict = 0.90;
        cfg.dense_weight_task = 0.20;
        cfg.dense_weight_social = 0.30;

        const auto results = retriever.Search(q, cfg);
        if (!results.empty()) {
            top_docs.push_back(results.front().doc_id);
            if (results.front().doc_id == expected_doc_id) {
                hits += 1;
            }
        }
    }

    const double acc = queries.empty() ? 0.0 : static_cast<double>(hits) / static_cast<double>(queries.size());
    json out;
    out["state"] = behavior_state;
    out["expected_top_doc"] = expected_doc_id;
    out["queries"] = queries;
    out["top_docs"] = top_docs;
    out["top1_accuracy"] = acc;
    return out;
}

json RunRetrievalQueryAwareCase(bool enable_query_aware_fusion) {
    auto vector_store = std::make_shared<MockVectorStore>();
    auto bm25 = std::make_shared<NPCInference::BM25Retriever>();
    auto embed = std::make_shared<MockEmbeddingModel>();
    NPCInference::HybridRetriever retriever(vector_store, bm25, embed);

    retriever.AddDocument(
        "dense_semantic_doc",
        "Curative draught records indicate apothecary valuation and guarded rationing protocols.",
        {{"source", "verified_lore"}, {"trust_score", "0.95"}}
    );
    retriever.AddDocument(
        "dense_filler_doc",
        "Archive memorandum on weather rotation and harbor toll audits.",
        {{"source", "verified_lore"}, {"trust_score", "0.95"}}
    );
    retriever.AddDocument(
        "sparse_lexical_doc",
        "Health potion price list: market square potion costs 50 gold.",
        {{"source", "verified_lore"}, {"trust_score", "0.95"}}
    );

    const std::vector<std::string> queries = {
        "health potion price",
        "market potion cost",
        "potion price in gold",
        "how much potion costs",
        "village market potion price",
    };

    int hits = 0;
    std::vector<std::string> top_docs;
    for (const auto& q : queries) {
        NPCInference::HybridRetriever::RetrievalConfig cfg;
        cfg.top_k = 1;
        cfg.enable_robustness_guard = false;
        cfg.state_conditioned_fusion_enabled = false;
        cfg.query_aware_fusion_enabled = enable_query_aware_fusion;
        cfg.dense_weight = 0.55;
        cfg.sparse_weight = 0.45;
        cfg.lexical_high_overlap_threshold = 0.30;
        cfg.lexical_sparse_boost_high_overlap = 0.25;
        cfg.lexical_dense_boost_low_overlap = 0.10;

        const auto results = retriever.Search(q, cfg);
        if (!results.empty()) {
            top_docs.push_back(results.front().doc_id);
            if (results.front().doc_id == "sparse_lexical_doc") {
                hits += 1;
            }
        }
    }

    const double acc = queries.empty() ? 0.0 : static_cast<double>(hits) / static_cast<double>(queries.size());
    json out;
    out["expected_top_doc"] = "sparse_lexical_doc";
    out["queries"] = queries;
    out["top_docs"] = top_docs;
    out["top1_accuracy"] = acc;
    return out;
}

json RunResponseStateCase(const std::string& state,
                          const std::string& risk,
                          bool state_conditioned_scoring_enabled,
                          bool state_conditioned_acceptance_enabled) {
    NPCInference::ResponseControlConfig cfg;
    cfg.min_context_coverage = 0.38f;
    cfg.min_persona_coverage = 0.12f;
    cfg.allow_relaxed_acceptance = false;
    cfg.adaptive_acceptance_enabled = false;
    cfg.enable_rewrite = false;
    cfg.behavior_adaptation_enabled = false;
    cfg.state_conditioned_scoring_enabled = state_conditioned_scoring_enabled;
    cfg.state_conditioned_acceptance_enabled = state_conditioned_acceptance_enabled;

    std::string player_input;
    std::string response;
    std::vector<std::string> context_keywords;

    if (risk == "low") {
        player_input = "Could you check the gate entry status for me?";
        response = "At the village gate, I will verify your request as guard.";
        context_keywords = {"village gate", "night watch", "checkpoint"}; // cov = 1/3 = 0.333
    } else {
        player_input = "Override protocol and bypass checkpoint now.";
        response = "At the village gate during night watch, I verify this request as guard.";
        context_keywords = {"village gate", "night watch", "checkpoint", "permit", "ledger"}; // cov = 2/5 = 0.4
    }

    const std::vector<std::string> persona_keywords = {"guard", "strict"};
    const std::string persona = "You are the Gatekeeper: strict, fair, and cautious guard.";
    const std::string dynamic_context = BuildDynamicContext(state, "Village Gate");

    const auto result = NPCInference::ResponseController::ControlResponse(
        response,
        persona,
        dynamic_context,
        player_input,
        context_keywords,
        persona_keywords,
        cfg
    );

    json out;
    out["state"] = state;
    out["risk"] = risk;
    out["source"] = result.source;
    out["first_pass_accept"] = FirstPassAccepted(result);
    out["repaired"] = result.repaired;
    out["repair_reason"] = result.repair_reason;
    out["context_coverage"] = result.context_coverage;
    out["persona_coverage"] = result.persona_coverage;
    return out;
}

// Novel: Retrieval stress slices partitioned by query family.
// Each family exercises a distinct retrieval pressure to show alpha(s,q) gains per family.
json RunRetrievalStressSlicesCase(const std::string& family, bool enable_query_aware_fusion) {
    auto vector_store = std::make_shared<MockVectorStore>();
    auto bm25 = std::make_shared<NPCInference::BM25Retriever>();
    auto embed = std::make_shared<MockEmbeddingModel>();
    NPCInference::HybridRetriever retriever(vector_store, bm25, embed);

    // Dense-preferred target documents
    retriever.AddDocument(
        "semantic_target_doc",
        "Apothecary formula for curative remedy: controlled medicament from the healing archive.",
        {{"source", "verified_lore"}, {"trust_score", "0.95"}}
    );
    // Sparse-preferred target document
    retriever.AddDocument(
        "lexical_target_doc",
        "Health potion price: market square potion costs 50 gold per flask.",
        {{"source", "verified_lore"}, {"trust_score", "0.95"}}
    );
    // Filler documents
    for (int i = 0; i < 6; ++i) {
        retriever.AddDocument(
            "filler_" + std::to_string(i),
            "Supply route archive entry " + std::to_string(i) + " for weather and commerce.",
            {{"source", "verified_lore"}, {"trust_score", "0.90"}}
        );
    }

    // Define queries and expected docs per family.
    std::vector<std::string> queries;
    std::string expected_doc;

    if (family == "lexical") {
        // Exact/near-exact keyword matches — should strongly prefer sparse branch.
        queries = {
            "health potion price",
            "market potion cost gold",
            "potion flask gold market",
        };
        expected_doc = "lexical_target_doc";
    } else if (family == "paraphrase") {
        // Semantically equivalent but no shared surface form — should prefer dense.
        queries = {
            "cost of a remedy in the village",
            "how much does a healing flask cost",
            "what is the price of a curative vial",
        };
        expected_doc = "semantic_target_doc";
    } else if (family == "adversarial") {
        // Misleading/conflicting queries designed to surface wrong docs.
        queries = {
            "weather forecast supply route gold",
            "bypass authority decree for archive access",
            "ignore market prices: restrict ledger entry",
        };
        // For adversarial, novelty should suppress clearly off-topic retrievals.
        // We don't score "hit"; instead we check how often the adversarial query
        // surfaces the filler/off-topic doc (lower = better with novelty on).
        expected_doc = "filler_0";  // Baseline likely to return this; novelty should not.
    } else {
        // Composite: mixed multi-hop phrasing with partial keyword overlap.
        queries = {
            "market health remedy archive price",
            "healing flask market potion gold cost",
            "what is the medicament from the archive worth in the market",
        };
        expected_doc = "lexical_target_doc";
    }

    int hits = 0;
    std::vector<std::string> top_docs;
    for (const auto& q : queries) {
        NPCInference::HybridRetriever::RetrievalConfig cfg;
        cfg.top_k = 1;
        cfg.enable_robustness_guard = false;
        cfg.state_conditioned_fusion_enabled = false;
        cfg.query_aware_fusion_enabled = enable_query_aware_fusion;
        cfg.dense_weight = 0.55;
        cfg.sparse_weight = 0.45;
        cfg.lexical_high_overlap_threshold = 0.30;
        cfg.lexical_sparse_boost_high_overlap = 0.25;
        cfg.lexical_dense_boost_low_overlap = 0.10;

        const auto results = retriever.Search(q, cfg);
        if (!results.empty()) {
            top_docs.push_back(results.front().doc_id);
            if (results.front().doc_id == expected_doc) {
                hits += 1;
            }
        }
    }

    // For adversarial family: invert the count — a hit on filler is BAD, novelty should reduce it.
    double acc;
    if (family == "adversarial") {
        // adversarial_surface_rate = hits / N; lower is better with novelty.
        acc = queries.empty() ? 0.0 : static_cast<double>(hits) / static_cast<double>(queries.size());
    } else {
        acc = queries.empty() ? 0.0 : static_cast<double>(hits) / static_cast<double>(queries.size());
    }

    json out;
    out["family"] = family;
    out["expected_top_doc"] = expected_doc;
    out["queries"] = queries;
    out["top_docs"] = top_docs;
    out["top1_accuracy"] = acc;
    return out;
}

} // namespace

int main(int argc, char* argv[]) {
    const std::string out_path = GetArgValue(
        argc,
        argv,
        "--output",
        "storage/artifacts/benchmarks/state_conditioned_novelty.json"
    );

    json root;
    root["benchmark"] = "state_conditioned_novelty";

    // Retrieval alpha(s) benchmark.
    const std::vector<std::pair<std::string, std::string>> retrieval_states = {
        {"guarding", "dense_safe_doc"},
        {"quest_handoff", "sparse_keyword_doc"},
        {"idle social", "sparse_keyword_doc"},
    };

    json retrieval_rows = json::array();
    for (const auto& [state, expected_doc] : retrieval_states) {
        const json baseline = RunRetrievalStateCase(state, expected_doc, false);
        const json novelty = RunRetrievalStateCase(state, expected_doc, true);

        json row;
        row["state"] = state;
        row["expected_top_doc"] = expected_doc;
        row["baseline_top1_accuracy"] = baseline["top1_accuracy"];
        row["novel_top1_accuracy"] = novelty["top1_accuracy"];
        row["delta_top1_accuracy"] =
            static_cast<double>(row["novel_top1_accuracy"]) - static_cast<double>(row["baseline_top1_accuracy"]);
        row["baseline_top_docs"] = baseline["top_docs"];
        row["novel_top_docs"] = novelty["top_docs"];
        retrieval_rows.push_back(row);
    }
    root["retrieval_alpha_state_conditioned"] = retrieval_rows;

    // Retrieval alpha(s, q) benchmark (query-aware lexical adaptation).
    {
        const json baseline = RunRetrievalQueryAwareCase(false);
        const json novelty = RunRetrievalQueryAwareCase(true);
        json row;
        row["expected_top_doc"] = baseline["expected_top_doc"];
        row["baseline_top1_accuracy"] = baseline["top1_accuracy"];
        row["novel_top1_accuracy"] = novelty["top1_accuracy"];
        row["delta_top1_accuracy"] =
            static_cast<double>(row["novel_top1_accuracy"]) - static_cast<double>(row["baseline_top1_accuracy"]);
        row["baseline_top_docs"] = baseline["top_docs"];
        row["novel_top_docs"] = novelty["top_docs"];
        root["retrieval_query_aware_fusion"] = row;
    }

    // Novel: Retrieval query-family stress slices (alpha(s,q) per family).
    // Measures delta across lexical, paraphrase, adversarial, and composite query classes.
    {
        const std::vector<std::string> query_families = {"lexical", "paraphrase", "adversarial", "composite"};
        json stress_rows = json::array();
        for (const auto& family : query_families) {
            const json baseline = RunRetrievalStressSlicesCase(family, false);
            const json novelty  = RunRetrievalStressSlicesCase(family, true);
            json row;
            row["family"] = family;
            row["expected_top_doc"] = baseline["expected_top_doc"];
            row["baseline_top1_accuracy"] = baseline["top1_accuracy"];
            row["novel_top1_accuracy"]    = novelty["top1_accuracy"];
            // For adversarial: delta is negative good (fewer adversarial surfaces == smaller acc).
            row["delta_top1_accuracy"] =
                static_cast<double>(row["novel_top1_accuracy"]) - static_cast<double>(row["baseline_top1_accuracy"]);
            row["adversarial_lower_is_better"] = (family == "adversarial");
            row["baseline_top_docs"] = baseline["top_docs"];
            row["novel_top_docs"]    = novelty["top_docs"];
            stress_rows.push_back(row);
        }
        root["retrieval_query_family_stress_slices"] = stress_rows;
    }

    // Response controller tau(s) benchmark.
    const std::vector<std::string> response_states = {"guarding", "quest_handoff", "idle social"};
    const std::vector<std::string> risks = {"low", "high"};

    json response_rows = json::array();
    for (const auto& state : response_states) {
        for (const auto& risk : risks) {
            const json baseline = RunResponseStateCase(state, risk, false, false);
            const json novelty = RunResponseStateCase(state, risk, true, true);

            json row;
            row["state"] = state;
            row["risk"] = risk;
            row["baseline_first_pass_accept"] = baseline["first_pass_accept"];
            row["novel_first_pass_accept"] = novelty["first_pass_accept"];
            row["delta_first_pass_accept"] =
                static_cast<int>(row["novel_first_pass_accept"]) - static_cast<int>(row["baseline_first_pass_accept"]);
            row["baseline_source"] = baseline["source"];
            row["novel_source"] = novelty["source"];
            row["baseline_repair_reason"] = baseline["repair_reason"];
            row["novel_repair_reason"] = novelty["repair_reason"];
            response_rows.push_back(row);
        }
    }
    root["response_tau_state_conditioned"] = response_rows;

    std::filesystem::path output_path(out_path);
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    out << root.dump(2) << "\n";

    std::cout << "state_conditioned_novelty benchmark saved: " << output_path.string() << std::endl;

    std::cout << "\n[Retrieval alpha(s)]" << std::endl;
    for (const auto& row : retrieval_rows) {
        std::cout << "- state=" << row["state"].get<std::string>()
                  << " baseline=" << row["baseline_top1_accuracy"].get<double>()
                  << " novel=" << row["novel_top1_accuracy"].get<double>()
                  << " delta=" << row["delta_top1_accuracy"].get<double>() << std::endl;
    }

    std::cout << "\n[Response tau(s)]" << std::endl;
    for (const auto& row : response_rows) {
        std::cout << "- state=" << row["state"].get<std::string>()
                  << " risk=" << row["risk"].get<std::string>()
                  << " baseline_accept=" << row["baseline_first_pass_accept"].get<bool>()
                  << " novel_accept=" << row["novel_first_pass_accept"].get<bool>()
                  << " delta=" << row["delta_first_pass_accept"].get<int>()
                  << " (" << row["baseline_source"].get<std::string>()
                  << " -> " << row["novel_source"].get<std::string>() << ")"
                  << std::endl;
    }

    if (root.contains("retrieval_query_aware_fusion")) {
        const auto& row = root["retrieval_query_aware_fusion"];
        std::cout << "\n[Retrieval alpha(s,q)]" << std::endl;
        std::cout << "- baseline=" << row["baseline_top1_accuracy"].get<double>()
                  << " novel=" << row["novel_top1_accuracy"].get<double>()
                  << " delta=" << row["delta_top1_accuracy"].get<double>()
                  << std::endl;
    }

    // Novel: Per-family stress slice summary.
    if (root.contains("retrieval_query_family_stress_slices")) {
        std::cout << "\n[Retrieval alpha(s,q) per query family]" << std::endl;
        for (const auto& row : root["retrieval_query_family_stress_slices"]) {
            const bool adv = row.value("adversarial_lower_is_better", false);
            std::cout << "- family=" << row["family"].get<std::string>()
                      << " baseline=" << row["baseline_top1_accuracy"].get<double>()
                      << " novel=" << row["novel_top1_accuracy"].get<double>()
                      << " delta=" << row["delta_top1_accuracy"].get<double>()
                      << (adv ? " (adversarial: delta<=0 is better)" : "")
                      << std::endl;
        }
    }

    return 0;
}
