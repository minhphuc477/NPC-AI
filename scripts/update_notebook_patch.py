import json
import textwrap

notebook_path = "notebooks/NPC_AI_Complete_Pipeline.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    updated = False
    
    # The new robust chat implementation with shared_ptr and context pruning
    new_chat_patch = """chat_new_source = textwrap.dedent(\"\"\"
    std::string NPCInferenceEngine::Chat(const std::string& session_id, const std::string& user_message) {
        if (!conversation_manager_) return \"Error: No conversation manager\";
        
        auto ctx = conversation_manager_->GetSession(session_id); // Now returns shared_ptr
        if (!ctx) return \"Error: Invalid session ID\";
        
        conversation_manager_->AddMessage(session_id, \"user\", user_message);
        json advanced_context = BuildAdvancedContext(ctx->npc_name, user_message);
        
        std::string history_str = \"\";
        auto history = conversation_manager_->GetHistory(session_id, 6);
        for (const auto& msg : history) {
            history_str += (msg.role == \"user\" ? ctx->player_name : ctx->npc_name) + \": \" + msg.content + \"\\\\n\";
        }
        
        advanced_context[\"recent_history\"] = history_str;
        advanced_context[\"npc_id\"] = ctx->npc_name;
        advanced_context[\"player_id\"] = ctx->player_name;
        advanced_context[\"conversation_id\"] = session_id;
        
        std::string response = GenerateWithState(user_message, advanced_context, false);
        conversation_manager_->AddMessage(session_id, \"assistant\", response);
        
        if (config_.enable_graph) Learn(user_message);
        return response;
    }
\"\"\")
with open('cpp/src/NPCInference.cpp', 'r', encoding='utf-8') as f: ni_content = f.read()
ni_content = re.sub(r'std::string NPCInferenceEngine::Chat.*?\\n\\s+\\}', chat_new_source, ni_content, flags=re.DOTALL)
with open('cpp/src/NPCInference.cpp', 'w', encoding='utf-8') as f: f.write(ni_content)
print('✅ Overwrote NPCInferenceEngine::Chat')
"""

    for cell in notebook.get('cells', []):
        if cell.get('id') == 'cpp_patch_cell':
            source = cell.get('source', [])
            
            # Find where Fix 5 starts
            start_idx = -1
            end_idx = -1
            for i, line in enumerate(source):
                if "# Fix 5: NPCInference.cpp" in line:
                    start_idx = i
                if "print('🎉 C++ patching complete!')" in line:
                    end_idx = i
            
            if start_idx != -1 and end_idx != -1:
                # Replace the lines for Fix 5
                # We need to format it back into an array of strings with newlines
                new_patch_lines = [line + '\n' for line in new_chat_patch.split('\n')]
                
                # Keep the header and footer
                new_source = source[:start_idx]
                new_source.append("# Fix 5: NPCInference.cpp — Wire BuildAdvancedContext into Chat() with shared_ptr\n")
                new_source.extend(new_patch_lines)
                new_source.append(source[end_idx]) # The print complete line
                
                cell['source'] = new_source
                updated = True
                print("Patched cpp_patch_cell in notebook.")

    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print("Could not find the target cpp_patch_cell.")

except Exception as e:
    print(f"Error updating notebook: {e}")
