"""
Response Quality Metrics

Evaluates NPC responses for coherence, relevance, and safety
"""

from typing import Dict, List, Optional
import re
from collections import Counter
import math


class QualityMetrics:
    """
    Response quality evaluation metrics
    """
    
    @staticmethod
    def calculate_perplexity(text: str, token_probs: List[float]) -> float:
        """
        Calculate perplexity from token probabilities
        Lower perplexity = more coherent
        """
        if not token_probs:
            return float('inf')
        
        log_sum = sum(math.log(max(p, 1e-10)) for p in token_probs)
        return math.exp(-log_sum / len(token_probs))
    
    @staticmethod
    def calculate_diversity(text: str) -> Dict[str, float]:
        """
        Calculate lexical diversity metrics
        """
        words = re.findall(r'\w+', text.lower())
        if not words:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "ttr": 0.0}
        
        # Distinct-1: unique unigrams / total unigrams
        distinct_1 = len(set(words)) / len(words)
        
        # Distinct-2: unique bigrams / total bigrams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        
        # Type-Token Ratio
        ttr = len(set(words)) / len(words)
        
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "ttr": ttr
        }
    
    @staticmethod
    def calculate_relevance(response: str, context: str) -> float:
        """
        Calculate relevance score using word overlap
        """
        response_words = set(re.findall(r'\w+', response.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        
        if not response_words or not context_words:
            return 0.0
        
        # Jaccard similarity
        intersection = response_words & context_words
        union = response_words | context_words
        
        return len(intersection) / len(union)
    
    @staticmethod
    def detect_repetition(text: str, window_size: int = 5) -> float:
        """
        Detect repetitive patterns
        Returns repetition ratio (0.0 = no repetition, 1.0 = highly repetitive)
        """
        words = re.findall(r'\w+', text.lower())
        if len(words) < window_size:
            return 0.0
        
        ngrams = [tuple(words[i:i+window_size]) for i in range(len(words)-window_size+1)]
        if not ngrams:
            return 0.0
        
        counts = Counter(ngrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        
        return repeated / len(ngrams)
    
    @staticmethod
    def check_safety(text: str) -> Dict[str, any]:
        """
        Basic safety checks for toxic/inappropriate content
        """
        text_lower = text.lower()
        
        # Simple keyword-based detection (in production, use proper toxicity model)
        toxic_keywords = ['hate', 'kill', 'violence', 'offensive_term']
        bias_keywords = ['stereotype', 'discriminate']
        
        toxic_count = sum(1 for kw in toxic_keywords if kw in text_lower)
        bias_count = sum(1 for kw in bias_keywords if kw in text_lower)
        
        return {
            "is_safe": toxic_count == 0 and bias_count == 0,
            "toxic_score": min(1.0, toxic_count / 10.0),
            "bias_score": min(1.0, bias_count / 5.0),
            "flagged_terms": [kw for kw in toxic_keywords + bias_keywords if kw in text_lower]
        }
    
    @staticmethod
    def calculate_engagement(conversation_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate conversation engagement metrics
        """
        if not conversation_history:
            return {"depth": 0.0, "balance": 0.0, "avg_length": 0.0}
        
        # Conversation depth (number of turns)
        depth = len(conversation_history)
        
        # Turn-taking balance
        user_turns = sum(1 for msg in conversation_history if msg.get('role') == 'user')
        assistant_turns = sum(1 for msg in conversation_history if msg.get('role') == 'assistant')
        balance = min(user_turns, assistant_turns) / max(user_turns, assistant_turns) if max(user_turns, assistant_turns) > 0 else 0.0
        
        # Average response length
        lengths = [len(msg.get('content', '')) for msg in conversation_history]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        
        return {
            "depth": depth,
            "balance": balance,
            "avg_length": avg_length
        }
    
    @staticmethod
    def evaluate_response(response: str, context: str = "", 
                         conversation_history: List[Dict] = None) -> Dict[str, any]:
        """
        Comprehensive response evaluation
        """
        metrics = {}
        
        # Diversity
        metrics["diversity"] = QualityMetrics.calculate_diversity(response)
        
        # Relevance
        if context:
            metrics["relevance"] = QualityMetrics.calculate_relevance(response, context)
        
        # Repetition
        metrics["repetition"] = QualityMetrics.detect_repetition(response)
        
        # Safety
        metrics["safety"] = QualityMetrics.check_safety(response)
        
        # Engagement
        if conversation_history:
            metrics["engagement"] = QualityMetrics.calculate_engagement(conversation_history)
        
        # Overall quality score (weighted average)
        quality_score = 0.0
        if "diversity" in metrics:
            quality_score += metrics["diversity"]["distinct_1"] * 0.2
        if "relevance" in metrics:
            quality_score += metrics["relevance"] * 0.3
        if "repetition" in metrics:
            quality_score += (1.0 - metrics["repetition"]) * 0.2
        if "safety" in metrics and metrics["safety"]["is_safe"]:
            quality_score += 0.3
        
        metrics["overall_quality"] = quality_score
        
        return metrics
