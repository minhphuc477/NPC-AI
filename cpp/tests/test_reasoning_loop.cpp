#include "NPCInference.h"
#include "ModelLoader.h"
#include "Tokenizer.h"
#include <iostream>
#include <cassert>
#include <vector>

namespace NPCInference {

// Mock for ModelLoader that simulates a multi-turn reasoning path
class MockReasoningModel : public ModelLoader {
public:
    int call_count = 0;

    virtual bool LoadModel(const std::string&, bool, int) override { return true; }
    virtual bool IsLoaded() const override { return true; }
    
    virtual std::vector<int64_t> Generate(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>&,
        int,
        const std::string&,
        std::function<void(int64_t)> callback,
        std::function<void(float*, int64_t)>
    ) override {
        call_count++;
        
        // Return IDs that MockTokenizer will decode into specific strings
        if (call_count == 1) {
            // First call: Return a tool call request
            return {100}; // "TOOL_CALL_MARKER"
        } else {
            // Second call: Return final response
            return {200}; // "FINAL_RESPONSE_MARKER"
        }
    }
};

class MockReasoningTokenizer : public Tokenizer {
public:
    virtual bool Load(const std::string&) override { return true; }
    virtual bool IsLoaded() const override { return true; }
    virtual std::vector<int64_t> Encode(const std::string&) override { return {1, 2, 3}; }
    virtual std::string Decode(const std::vector<int64_t>& ids) override {
        if (ids.empty()) return "";
        if (ids.size() > 0 && ids.back() == 100) return "I need to check the inventory first. [TOOL: check_inventory(\"potions\")]";
        if (ids.size() > 0 && ids.back() == 200) return "I have 5 potions left.";
        return "Unknown";
    }
    virtual int GetVocabSize() const override { return 1000; }
    virtual int GetEOSId() const override { return 999; }
};

class TestableReasoningEngine : public NPCInferenceEngine {
public:
    TestableReasoningEngine() {
        model_loader_.reset(new MockReasoningModel());
        tokenizer_.reset(new MockReasoningTokenizer());
        prompt_builder_ = std::make_unique<PromptBuilder>(true);
        tool_registry_ = std::make_unique<ToolRegistry>();
        
        // Register a mock tool
        tool_registry_->RegisterTool("check_inventory", 
                                   "Check current inventory",
                                   nlohmann::json::object(),
                                   [](const nlohmann::json& args) {
            return "Inventory: 5 potions found.";
        });

        ready_ = true;
    }

    int GetModelCallCount() {
        return static_cast<MockReasoningModel*>(model_loader_.get())->call_count;
    }
};

} // namespace NPCInference

int main() {
    using namespace NPCInference;
    
    std::cout << "Starting Recursive Reasoning Loop Test..." << std::endl;
    
    TestableReasoningEngine engine;
    
    nlohmann::json state;
    state["npc_id"] = "Tester";
    state["tools"] = nlohmann::json::array({ { "name", "check_inventory", "description", "Check items" } });
    
    std::cout << "Turn 1: Testing generation with tool call deduction..." << std::endl;
    // Note: We need to use local_state that contains the tools
    std::string response = engine.GenerateWithState("How many potions do I have?", state);
    
    std::cout << "Final Response: " << response << std::endl;
    
    // VERIFICATIONS
    // 1. Model should have been called twice (Turn 1 -> Tool -> Turn 2)
    int count = engine.GetModelCallCount();
    std::cout << "Model call count: " << count << std::endl;
    assert(count == 2);
    std::cout << "✓ Logic verified: Model called exactly 2 times." << std::endl;
    
    // 2. Final response should be the second model output
    assert(response == "I have 5 potions left.");
    std::cout << "✓ Logic verified: Final response matches expected refined output." << std::endl;
    
    std::cout << "Recursive Reasoning Loop Test PASSED." << std::endl;
    return 0;
}
