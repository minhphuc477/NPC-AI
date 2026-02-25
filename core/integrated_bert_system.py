"""
Integrated BERT-based NPC Evaluation System

Combines all advanced features:
- GPU-accelerated BERT embeddings
- Multi-language support
- Emotion-aware scoring
- Player satisfaction correlation
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

from .bert_evaluator import BERTSemanticEvaluator, SemanticScore
from .multilanguage_support import MultiLanguageSupport
from .emotion_scoring import EmotionAwareScoring, EmotionProfile


@dataclass
class ComprehensiveScore:
    """Comprehensive evaluation score"""
    # Semantic metrics
    semantic_similarity: float
    contextual_relevance: float
    coherence: float
    
    # Emotion metrics
    emotion: str
    emotion_confidence: float
    emotional_appropriateness: float
    
    # Language metrics
    language: str
    language_confidence: float
    
    # Overall
    overall_quality: float
    
    # Player satisfaction (predicted)
    predicted_satisfaction: float


class IntegratedBERTSystem:
    """
    Integrated BERT-based evaluation system
    
    Combines:
    - BERT semantic evaluation
    - Multi-language support
    - Emotion-aware scoring
    - Player satisfaction prediction
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        default_language: str = 'en'
    ):
        """
        Initialize integrated system
        
        Args:
            use_gpu: Whether to use GPU acceleration
            default_language: Default language code
        """
        self.bert_evaluator = BERTSemanticEvaluator(use_gpu=use_gpu)
        self.multilang = MultiLanguageSupport(default_language=default_language)
        self.emotion_scorer = EmotionAwareScoring(use_gpu=use_gpu)
        
        # Baseline linear satisfaction model (replace with learned model when labeled data is available).
        self._satisfaction_weights = {
            'semantic': 0.25,
            'emotion': 0.30,
            'relevance': 0.25,
            'coherence': 0.20
        }
    
    def evaluate_response(
        self,
        response: str,
        context: str = "",
        context_type: Optional[str] = None,
        reference: Optional[str] = None,
        detect_language: bool = True
    ) -> ComprehensiveScore:
        """
        Comprehensive evaluation of NPC response
        
        Args:
            response: NPC response text
            context: Conversation context
            context_type: Type of context (combat, greeting, etc.)
            reference: Optional reference response
            detect_language: Whether to detect language
            
        Returns:
            ComprehensiveScore with all metrics
        """
        # 1. Language detection
        if detect_language:
            language = self.multilang.detect_language(response)
            language_confidence = 0.9  # Placeholder
        else:
            language = self.multilang.default_language
            language_confidence = 1.0
        
        # 2. BERT semantic evaluation
        semantic_score = self.bert_evaluator.evaluate_response(
            response,
            context=context,
            reference=reference,
            detect_emotion=False  # We'll use our emotion scorer
        )
        
        # 3. Emotion analysis
        emotion_profile = self.emotion_scorer.analyze_emotion(response)
        
        # 4. Emotional appropriateness
        if context_type:
            emotional_appropriateness = self.emotion_scorer.score_emotional_appropriateness(
                emotion_profile,
                context_type
            )
        else:
            emotional_appropriateness = 0.5
        
        # 5. Predict player satisfaction
        predicted_satisfaction = self._predict_satisfaction(
            semantic_score,
            emotion_profile,
            emotional_appropriateness
        )
        
        return ComprehensiveScore(
            semantic_similarity=semantic_score.similarity,
            contextual_relevance=semantic_score.relevance,
            coherence=semantic_score.coherence,
            emotion=emotion_profile.primary_emotion,
            emotion_confidence=emotion_profile.confidence,
            emotional_appropriateness=emotional_appropriateness,
            language=language,
            language_confidence=language_confidence,
            overall_quality=semantic_score.overall,
            predicted_satisfaction=predicted_satisfaction
        )
    
    def _predict_satisfaction(
        self,
        semantic_score: SemanticScore,
        emotion_profile: EmotionProfile,
        emotional_appropriateness: float
    ) -> float:
        """
        Predict player satisfaction based on metrics
        
        Args:
            semantic_score: Semantic evaluation
            emotion_profile: Emotion analysis
            emotional_appropriateness: Emotion appropriateness score
            
        Returns:
            Predicted satisfaction (0-1)
        """
        # Weighted combination of factors
        satisfaction = (
            semantic_score.overall * self._satisfaction_weights['semantic'] +
            emotion_profile.confidence * self._satisfaction_weights['emotion'] +
            semantic_score.relevance * self._satisfaction_weights['relevance'] +
            semantic_score.coherence * self._satisfaction_weights['coherence']
        )
        
        # Boost for appropriate emotions
        satisfaction += emotional_appropriateness * 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, satisfaction))
    
    def batch_evaluate(
        self,
        responses: List[str],
        contexts: List[str],
        context_types: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> List[ComprehensiveScore]:
        """
        Batch evaluate multiple responses
        
        Args:
            responses: List of responses
            contexts: List of contexts
            context_types: Optional list of context types
            batch_size: Batch size for processing
            
        Returns:
            List of ComprehensiveScores
        """
        if context_types is None:
            context_types = [None] * len(responses)
        
        results = []
        for i in range(0, len(responses), batch_size):
            batch_responses = responses[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            batch_types = context_types[i:i+batch_size]
            
            for response, context, ctx_type in zip(batch_responses, batch_contexts, batch_types):
                score = self.evaluate_response(response, context, ctx_type)
                results.append(score)
        
        return results
    
    def get_feedback(
        self,
        score: ComprehensiveScore,
        context_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get actionable feedback based on scores
        
        Args:
            score: Comprehensive score
            context_type: Optional context type
            
        Returns:
            Feedback dictionary
        """
        feedback = {
            'overall_quality': score.overall_quality,
            'predicted_satisfaction': score.predicted_satisfaction,
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        # Identify strengths
        if score.coherence > 0.7:
            feedback['strengths'].append("High coherence")
        if score.emotion_confidence > 0.7:
            feedback['strengths'].append(f"Clear {score.emotion} emotion")
        if score.contextual_relevance > 0.7:
            feedback['strengths'].append("Highly relevant to context")
        
        # Identify areas for improvement
        if score.coherence < 0.5:
            feedback['improvements'].append("Improve coherence between sentences")
        if score.emotional_appropriateness < 0.5:
            feedback['improvements'].append(f"Emotion ({score.emotion}) may not fit context")
        if score.contextual_relevance < 0.5:
            feedback['improvements'].append("Increase relevance to context")
        
        # Provide suggestions
        if score.predicted_satisfaction < 0.6:
            feedback['suggestions'].append("Consider revising response for better player engagement")
        if score.emotional_appropriateness < 0.5 and context_type:
            appropriate = self.emotion_scorer.CONTEXT_EMOTIONS.get(context_type, [])
            feedback['suggestions'].append(f"Try emotions: {', '.join(appropriate)}")
        
        return feedback
    
    def export_metrics(
        self,
        scores: List[ComprehensiveScore],
        filepath: str
    ):
        """
        Export metrics to JSON file
        
        Args:
            scores: List of scores
            filepath: Output file path
        """
        data = {
            'scores': [asdict(score) for score in scores],
            'summary': {
                'count': len(scores),
                'avg_quality': sum(s.overall_quality for s in scores) / len(scores),
                'avg_satisfaction': sum(s.predicted_satisfaction for s in scores) / len(scores),
                'avg_relevance': sum(s.contextual_relevance for s in scores) / len(scores)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Metrics exported to: {filepath}")


# Convenience functions
def quick_evaluate(
    response: str,
    context: str = "",
    context_type: Optional[str] = None,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Quick evaluation with simple output
    
    Args:
        response: NPC response
        context: Conversation context
        context_type: Context type
        use_gpu: Use GPU acceleration
        
    Returns:
        Simple score dictionary
    """
    system = IntegratedBERTSystem(use_gpu=use_gpu)
    score = system.evaluate_response(response, context, context_type)
    
    return {
        'overall_quality': score.overall_quality,
        'predicted_satisfaction': score.predicted_satisfaction,
        'emotion': score.emotion,
        'language': score.language,
        'relevance': score.contextual_relevance
    }


if __name__ == "__main__":
    # Example usage
    print("Integrated BERT System - Example")
    print("="*60)
    
    print("\nInitializing system...")
    print("(Note: Requires BERT models to be installed)")
    
    # Example evaluation
    response = "Greetings brave warrior! I have a dangerous quest for you."
    context = "Player approaches quest giver"
    context_type = "quest_offer"
    
    print(f"\nResponse: {response}")
    print(f"Context: {context}")
    print(f"Type: {context_type}")
    
    # Uncomment when dependencies installed:
    # system = IntegratedBERTSystem(use_gpu=True)
    # score = system.evaluate_response(response, context, context_type)
    # feedback = system.get_feedback(score, context_type)
    # 
    # print(f"\nResults:")
    # print(f"  Overall Quality: {score.overall_quality:.3f}")
    # print(f"  Predicted Satisfaction: {score.predicted_satisfaction:.3f}")
    # print(f"  Emotion: {score.emotion} ({score.emotion_confidence:.3f})")
    # print(f"  Language: {score.language}")
    # print(f"\nFeedback:")
    # print(f"  Strengths: {feedback['strengths']}")
    # print(f"  Improvements: {feedback['improvements']}")
