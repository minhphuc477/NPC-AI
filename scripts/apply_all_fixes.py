#!/usr/bin/env python3
"""
Comprehensive fix script to apply all critical fixes from harsh audit.
This script automates the fixes for issues that can be safely automated.
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodebaseFixer:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.fixes_applied = []
    
    def fix_bare_except_clauses(self):
        """Fix Issue #4: Replace bare except: with except Exception:"""
        logger.info("Fixing bare except clauses...")
        
        files_to_fix = [
            "scripts/ultra_low_mem_merge.py",
            "scripts/manual_weight_merge.py",
            "scripts/extreme_sharded_merge.py"
        ]
        
        for file_path in files_to_fix:
            full_path = self.root / file_path
            if not full_path.exists():
                continue
            
            content = full_path.read_text(encoding='utf-8')
            original = content
            
            # Replace bare except: with except Exception as e:
            content = re.sub(
                r'except:\s*$',
                'except Exception as e:',
                content,
                flags=re.MULTILINE
            )
            
            # Replace except: pass with proper handling
            content = re.sub(
                r'except:\s*pass',
                'except Exception as e:\\n        logger.warning(f"Error: {e}")\\n        pass',
                content
            )
            
            if content != original:
                full_path.write_text(content, encoding='utf-8')
                self.fixes_applied.append(f"Fixed bare except in {file_path}")
                logger.info(f"✓ Fixed {file_path}")
    
    def add_error_logging_to_cpp(self):
        """Fix Issue #2: Add logging to catch-all handlers"""
        logger.info("Adding error logging to C++ catch-all handlers...")
        
        cpp_files = list((self.root / "cpp" / "src").glob("*.cpp"))
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text(encoding='utf-8')
            original = content
            
            # Replace } catch (...) {} with proper logging
            content = re.sub(
                r'}\s*catch\s*\(\.\.\.\)\s*{}',
                '''} catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
    }''',
                content
            )
            
            # Replace } catch (...) { return false; } with logging
            content = re.sub(
                r'}\s*catch\s*\(\.\.\.\)\s*{\s*return\s+false;\s*}',
                '''} catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return false;
    }''',
                content
            )
            
            if content != original:
                cpp_file.write_text(content, encoding='utf-8')
                self.fixes_applied.append(f"Added error logging to {cpp_file.name}")
                logger.info(f"✓ Fixed {cpp_file.name}")
    
    def create_constants_header(self):
        """Fix Issue #8: Create constants header for magic numbers"""
        logger.info("Creating constants header...")
        
        constants_content = """#pragma once

// NPC AI Constants
// This file contains all magic numbers extracted from the codebase

namespace NPCInference {
namespace Constants {

// Token IDs
constexpr int64_t PHI3_EOS_TOKEN = 32000;
constexpr int64_t PHI3_END_TOKEN = 32007;

// Cache Configuration
constexpr size_t DEFAULT_CACHE_SIZE_MB = 512;
constexpr size_t MAX_CONVERSATIONS = 100;

// Generation Limits
constexpr int DEFAULT_MAX_TOKENS = 150;
constexpr int PLANNING_MAX_TOKENS = 100;
constexpr int CRITIQUE_MAX_TOKENS = 150;
constexpr int REFINE_MAX_TOKENS = 150;
constexpr int TRUTH_CHECK_MAX_TOKENS = 100;

// RAG Configuration
constexpr int DEFAULT_RAG_TOP_K = 3;
constexpr int HYBRID_RAG_TOP_K = 5;
constexpr float DEFAULT_RAG_THRESHOLD = 0.7f;

// Memory Configuration
constexpr float MEMORY_IMPORTANCE_THRESHOLD = 0.3f;
constexpr size_t MIN_MEMORIES_FOR_CONSOLIDATION = 5;
constexpr size_t CONVERSATION_HISTORY_LIMIT = 6;

// Embedding Dimensions
constexpr size_t MINILM_EMBEDDING_DIM = 384;

// Input Validation
constexpr size_t MAX_INPUT_LENGTH = 1000;
constexpr size_t MAX_PROMPT_LENGTH = 4096;

// Timeouts (milliseconds)
constexpr int OLLAMA_TIMEOUT_MS = 30000;
constexpr int MOCK_LATENCY_MS = 50;

} // namespace Constants
} // namespace NPCInference
"""
        
        constants_file = self.root / "cpp" / "include" / "Constants.h"
        constants_file.parent.mkdir(exist_ok=True, parents=True)
        constants_file.write_text(constants_content, encoding='utf-8')
        self.fixes_applied.append("Created Constants.h")
        logger.info("✓ Created Constants.h")
    
    def remove_debug_prints(self):
        """Fix Issue #14: Remove debug prints from production code"""
        logger.info("Removing debug prints...")
        
        cpp_files = list((self.root / "cpp" / "src").glob("*.cpp"))
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text(encoding='utf-8')
            original = content
            
            # Remove DEBUG: prints
            content = re.sub(
                r'std::cout\s*<<\s*"DEBUG:.*?<<\s*std::endl;\s*\n',
                '',
                content
            )
            
            if content != original:
                cpp_file.write_text(content, encoding='utf-8')
                self.fixes_applied.append(f"Removed debug prints from {cpp_file.name}")
                logger.info(f"✓ Cleaned {cpp_file.name}")
    
    def remove_commented_code(self):
        """Fix Issue #11: Remove commented code pollution"""
        logger.info("Removing commented code blocks...")
        
        cpp_files = list((self.root / "cpp" / "src").glob("*.cpp"))
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text(encoding='utf-8')
            original = content
            
            # Remove large commented blocks (10+ lines)
            lines = content.split('\\n')
            cleaned_lines = []
            comment_block = []
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('/*') or stripped.startswith('//'):
                    comment_block.append(line)
                else:
                    if len(comment_block) > 10:
                        # Large comment block, likely dead code
                        pass  # Skip it
                    else:
                        cleaned_lines.extend(comment_block)
                    comment_block = []
                    cleaned_lines.append(line)
            
            content = '\\n'.join(cleaned_lines)
            
            if content != original:
                cpp_file.write_text(content, encoding='utf-8')
                self.fixes_applied.append(f"Removed commented code from {cpp_file.name}")
                logger.info(f"✓ Cleaned {cpp_file.name}")
    
    def generate_fix_summary(self):
        """Generate summary of all fixes applied"""
        summary = f"""
# Automated Fixes Applied

Total fixes: {len(self.fixes_applied)}

## Fixes Applied:
"""
        for fix in self.fixes_applied:
            summary += f"- {fix}\\n"
        
        summary += """

## Manual Fixes Still Required:

### Critical:
1. **Remove const_cast** - Requires understanding ONNX Runtime API
   - Files: ModelLoader.cpp:226, 389; VisionLoader.cpp:156
   - Action: Use mutable data or copy before passing to ONNX

2. **Add thread safety** - Requires mutex implementation
   - File: NPCInference.cpp
   - Action: Add std::mutex for current_state_, current_action_, last_thought_

3. **Add input validation** - Requires domain knowledge
   - File: chat_interface.cpp, NPCInference.cpp
   - Action: Validate length, sanitize special characters

### High Priority:
4. **Fix memory leak potential** - Requires careful review
   - File: ModelLoader.cpp
   - Action: Ensure exception safety in KV cache moves

5. **Add bounds checking** - Requires understanding of tensor shapes
   - File: ModelLoader.cpp:454, 705
   - Action: Check indices before array access

## Next Steps:
1. Review this summary
2. Test automated fixes
3. Apply manual fixes
4. Run full test suite
5. Verify no regressions
"""
        
        summary_file = self.root.parent / ".gemini" / "antigravity" / "brain" / "b8445dba-3490-4652-bb82-385cc0e1ca0e" / "automated_fixes_summary.md"
        summary_file.parent.mkdir(exist_ok=True, parents=True)
        summary_file.write_text(summary, encoding='utf-8')
        logger.info(f"✓ Generated fix summary: {summary_file}")
        
        return summary


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python apply_all_fixes.py <project_root>")
        sys.exit(1)
    
    project_root = sys.argv[1]
    fixer = CodebaseFixer(project_root)
    
    logger.info("=" * 60)
    logger.info("Applying All Audit Fixes")
    logger.info("=" * 60)
    
    # Apply automated fixes
    fixer.fix_bare_except_clauses()
    fixer.add_error_logging_to_cpp()
    fixer.create_constants_header()
    fixer.remove_debug_prints()
    fixer.remove_commented_code()
    
    # Generate summary
    summary = fixer.generate_fix_summary()
    
    logger.info("=" * 60)
    logger.info(f"Applied {len(fixer.fixes_applied)} automated fixes")
    logger.info("See automated_fixes_summary.md for details")
    logger.info("=" * 60)
    
    print(summary)


if __name__ == "__main__":
    main()
