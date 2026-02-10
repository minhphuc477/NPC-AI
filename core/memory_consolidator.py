"""
LLM-Powered Memory Consolidation

Uses LLM to intelligently summarize and compress memories
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class MemoryConsolidator:
    """
    LLM-powered memory consolidation and summarization
    """
    
    def __init__(self, llm_generator=None):
        """
        Args:
            llm_generator: Function that takes prompt and returns LLM response
        """
        self.llm_generator = llm_generator
    
    def summarize_conversation(self, messages: List[Dict[str, str]], 
                               max_length: int = 200) -> str:
        """
        Summarize a conversation using LLM
        
        Args:
            messages: List of {role, content} dicts
            max_length: Maximum summary length in tokens
        
        Returns:
            Summary string
        """
        if not self.llm_generator:
            # Fallback to simple concatenation
            return self._simple_summarize(messages, max_length)
        
        # Build prompt for LLM
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])
        
        prompt = f"""Summarize the following conversation concisely, preserving key facts, decisions, and emotional tone:

{conversation_text}

Summary (max {max_length} words):"""
        
        try:
            summary = self.llm_generator(prompt, max_tokens=max_length)
            return summary.strip()
        except Exception as e:
            print(f"LLM summarization failed: {e}, using fallback")
            return self._simple_summarize(messages, max_length)
    
    def _simple_summarize(self, messages: List[Dict[str, str]], max_length: int) -> str:
        """Simple rule-based summarization fallback"""
        if not messages:
            return ""
        
        # Extract key information
        participants = set()
        topics = []
        
        for msg in messages:
            if msg['role'] != 'system':
                participants.add(msg['role'])
            
            # Simple topic extraction (first few words)
            content = msg['content']
            if len(content) > 20:
                topics.append(content[:50] + "...")
        
        summary = f"Conversation between {', '.join(participants)}. "
        summary += f"Discussed: {'; '.join(topics[:3])}"
        
        return summary[:max_length]
    
    def extract_facts(self, text: str) -> List[str]:
        """
        Extract factual statements from text
        
        Args:
            text: Input text
        
        Returns:
            List of extracted facts
        """
        if not self.llm_generator:
            return self._simple_extract_facts(text)
        
        prompt = f"""Extract key facts and information from the following text as a JSON list:

{text}

Facts (JSON array of strings):"""
        
        try:
            response = self.llm_generator(prompt, max_tokens=300)
            facts = json.loads(response)
            return facts if isinstance(facts, list) else []
        except Exception as e:
            print(f"Fact extraction failed: {e}, using fallback")
            return self._simple_extract_facts(text)
    
    def _simple_extract_facts(self, text: str) -> List[str]:
        """Simple rule-based fact extraction"""
        # Split into sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        # Filter for factual-looking sentences
        facts = []
        keywords = ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'can']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and any(kw in sentence.lower() for kw in keywords):
                facts.append(sentence)
        
        return facts[:10]  # Limit to top 10
    
    def consolidate_similar_memories(self, memories: List[str]) -> str:
        """
        Consolidate multiple similar memories into one
        
        Args:
            memories: List of memory strings
        
        Returns:
            Consolidated memory
        """
        if len(memories) == 1:
            return memories[0]
        
        if not self.llm_generator:
            # Simple concatenation
            return " | ".join(memories[:3])
        
        memories_text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(memories)])
        
        prompt = f"""Consolidate the following related memories into a single, comprehensive memory:

{memories_text}

Consolidated memory:"""
        
        try:
            consolidated = self.llm_generator(prompt, max_tokens=200)
            return consolidated.strip()
        except Exception as e:
            print(f"Memory consolidation failed: {e}")
            return " | ".join(memories[:3])
    
    def assess_importance(self, memory_text: str, context: str = "") -> float:
        """
        Assess importance of a memory
        
        Args:
            memory_text: The memory content
            context: Additional context
        
        Returns:
            Importance score (0.0 to 1.0)
        """
        if not self.llm_generator:
            return self._simple_importance(memory_text)
        
        prompt = f"""Rate the importance of this memory on a scale of 0.0 to 1.0:

Memory: {memory_text}
Context: {context}

Importance score (0.0-1.0):"""
        
        try:
            response = self.llm_generator(prompt, max_tokens=10)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Importance assessment failed: {e}")
            return self._simple_importance(memory_text)
    
    def _simple_importance(self, text: str) -> float:
        """Simple heuristic importance scoring"""
        score = 0.5  # Base score
        
        # Length factor
        if len(text) > 100:
            score += 0.1
        
        # Keyword boosting
        important_keywords = ['important', 'critical', 'remember', 'never', 'always', 
                             'quest', 'mission', 'secret', 'danger']
        for keyword in important_keywords:
            if keyword in text.lower():
                score += 0.1
        
        # Emotional keywords
        emotional_keywords = ['love', 'hate', 'fear', 'angry', 'happy', 'sad']
        for keyword in emotional_keywords:
            if keyword in text.lower():
                score += 0.05
        
        return min(1.0, score)
