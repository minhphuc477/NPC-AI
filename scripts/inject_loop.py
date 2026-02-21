import re
import os

filepath = r"f:\NPC AI\cpp\src\NPCInference.cpp"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# The block to replace (starts with building prompt and ends with final_output assignment)
# I'll use a regex to be flexible with whitespace
pattern = re.compile(r'// 2\. Build Prompt.*?final_output = tokenizer_->Decode\(new_tokens\);', re.DOTALL)

replacement = """// 2. Build Prompt (MULTI-TURN REASONING LOOP)
                std::string current_prompt = prompt;
                std::string final_output = "";
                
                for (int reasoning_step = 0; reasoning_step < 3; ++reasoning_step) {
                    std::string full_prompt;
                    std::string lang = local_state.is_object() ? local_state.value("language", "vi") : "vi";
                    nlohmann::json tools = (local_state.is_object() && local_state.contains("tools")) ? local_state["tools"] : nlohmann::json::array();
                    
                    if (config_.enable_planner && !is_json) {
                        // Inherit last_thought_ from previous turn if available
                        full_prompt = prompt_builder_->BuildWithThought(local_state, local_state, current_prompt, last_thought_, lang, tools);
                    } else {
                        try {
                            full_prompt = prompt_builder_->Build(local_state, local_state, current_prompt, lang, tools);
                        } catch (const std::exception& e) {
                            std::cerr << "PromptBuilder Error: " << e.what() << std::endl;
                            full_prompt = current_prompt; 
                        }
                    }
                    
                    std::string conv_id = local_state.is_object() ? local_state.value("conversation_id", local_state.value("npc_id", "default")) : "default";
                    std::vector<int64_t> input_ids = tokenizer_->Encode(full_prompt);
                    if (input_ids.empty()) break;

                    std::unique_ptr<GrammarSampler> local_sampler;
                    if (config_.enable_grammar && (is_json || reasoning_step < 2)) {
                        local_sampler = std::make_unique<GrammarSampler>(tokenizer_.get());
                        local_sampler->Reset();
                    }

                    auto logit_processor = [&](float* logits, int64_t vocab_size) {
                        if (local_sampler) local_sampler->FilterLogits(logits, vocab_size);
                    };
                    auto token_callback = [&](int64_t token) {
                        if (local_sampler) local_sampler->AcceptToken(token);
                    };

                    std::vector<int64_t> output_ids;
                    std::vector<int64_t> attention_mask(input_ids.size(), 1);
                    
                    output_ids = model_loader_->Generate(input_ids, attention_mask, 150, conv_id + "_step_" + std::to_string(reasoning_step), token_callback, logit_processor);
                    
                    if (output_ids.size() > input_ids.size()) {
                        std::vector<int64_t> new_tokens(output_ids.begin() + input_ids.size(), output_ids.end());
                        std::string step_output = tokenizer_->Decode(new_tokens);
                        
                        GenerationResult parse_res = ParseOutput(step_output);
                        if (!parse_res.tool_call.empty()) {
                            std::cout << "Reasoning Step " << reasoning_step << ": Executing Tool..." << std::endl;
                            std::string observation = ExecuteAction(parse_res.tool_call);
                            
                            json tool_obs;
                            tool_obs["step"] = reasoning_step;
                            tool_obs["call"] = parse_res.tool_call;
                            tool_obs["observation"] = observation;
                            local_state["tool_results"].push_back(tool_obs);
                            
                            current_prompt = prompt; 
                            continue; 
                        }

                        final_output = step_output;
                        profiler_->RecordTokens(new_tokens.size());
                        break;
                    } else break;
                }"""

new_content = pattern.sub(replacement, content, count=1)

if new_content != content:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✓ Successfully injected Reasoning Loop into NPCInference.cpp")
else:
    print("✗ Failed to find the target block in NPCInference.cpp")
