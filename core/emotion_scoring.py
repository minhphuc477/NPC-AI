"""
Emotion-Aware Semantic Scoring

Analyzes emotional content and adjusts semantic scores accordingly.
Helps create more emotionally appropriate NPC responses.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EmotionProfile:
    """Emotion analysis profile"""
    primary_emotion: str
    confidence: float
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    emotions: Dict[str, float]  # All emotion scores


class EmotionAwareScoring:
    """
    Emotion-aware semantic scoring system
    
    Features:
    - Emotion detection in responses
    - Emotional appropriateness scoring
    - Emotion-context matching
    - Emotional consistency tracking
    """
    
    # Emotion categories with valence and arousal
    EMOTION_PROFILES = {
        'joy': {'valence': 0.8, 'arousal': 0.7},
        'sadness': {'valence': -0.7, 'arousal': 0.3},
        'anger': {'valence': -0.6, 'arousal': 0.9},
        'fear': {'valence': -0.8, 'arousal': 0.8},
        'surprise': {'valence': 0.0, 'arousal': 0.8},
        'disgust': {'valence': -0.7, 'arousal': 0.5},
        'neutral': {'valence': 0.0, 'arousal': 0.0},
        'excitement': {'valence': 0.9, 'arousal': 0.9},
        'gratitude': {'valence': 0.7, 'arousal': 0.4},
        'pride': {'valence': 0.6, 'arousal': 0.5}
    }
    
    # Context-appropriate emotions for different scenarios
    CONTEXT_EMOTIONS = {
        'combat': ['anger', 'fear', 'excitement'],
        'greeting': ['joy', 'neutral', 'surprise'],
        'quest_offer': ['excitement', 'pride', 'neutral'],
        'merchant': ['joy', 'neutral', 'gratitude'],
        'sad_news': ['sadness', 'fear', 'neutral'],
        'celebration': ['joy', 'excitement', 'gratitude'],
        'threat': ['anger', 'fear', 'disgust']
    }
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize emotion-aware scoring
        
        Args:
            use_gpu: Whether to use GPU for emotion detection
        """
        self.use_gpu = use_gpu
        self._emotion_model = None
    
    def _load_emotion_model(self):
        """Lazy load emotion detection model"""
        if self._emotion_model is None:
            try:
                from transformers import pipeline
                import torch
                
                device = 0 if self.use_gpu and torch.cuda.is_available() else -1
                
                self._emotion_model = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=device,
                    top_k=None
                )
                print("✓ Loaded emotion detection model")
            except ImportError:
                print("⚠ transformers not installed")
                raise
        
        return self._emotion_model
    
    def analyze_emotion(self, text: str) -> EmotionProfile:
        """
        Analyze emotional content of text
        
        Args:
            text: Text to analyze
            
        Returns:
            EmotionProfile with detailed analysis
        """
        model = self._load_emotion_model()
        
        # Get emotion predictions
        results = model(text)[0]
        
        # Convert to dict
        emotions = {r['label']: r['score'] for r in results}
        
        # Get primary emotion
        primary = max(emotions.items(), key=lambda x: x[1])
        primary_emotion = primary[0]
        confidence = primary[1]
        
        # Calculate valence and arousal
        valence = 0.0
        arousal = 0.0
        
        for emotion, score in emotions.items():
            if emotion in self.EMOTION_PROFILES:
                profile = self.EMOTION_PROFILES[emotion]
                valence += profile['valence'] * score
                arousal += profile['arousal'] * score
        
        return EmotionProfile(
            primary_emotion=primary_emotion,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            emotions=emotions
        )
    
    def score_emotional_appropriateness(
        self,
        response_emotion: EmotionProfile,
        context_type: str
    ) -> float:
        """
        Score how appropriate the emotion is for the context
        
        Args:
            response_emotion: Detected emotion in response
            context_type: Type of context (combat, greeting, etc.)
            
        Returns:
            Appropriateness score (0-1)
        """
        if context_type not in self.CONTEXT_EMOTIONS:
            return 0.5  # Neutral if unknown context
        
        appropriate_emotions = self.CONTEXT_EMOTIONS[context_type]
        
        # Check if primary emotion is appropriate
        if response_emotion.primary_emotion in appropriate_emotions:
            return response_emotion.confidence
        
        # Check if any appropriate emotion has significant score
        max_appropriate = 0.0
        for emotion in appropriate_emotions:
            if emotion in response_emotion.emotions:
                max_appropriate = max(max_appropriate, response_emotion.emotions[emotion])
        
        return max_appropriate
    
    def calculate_emotional_consistency(
        self,
        emotions: List[EmotionProfile]
    ) -> float:
        """
        Calculate emotional consistency across multiple responses
        
        Args:
            emotions: List of emotion profiles
            
        Returns:
            Consistency score (0-1)
        """
        if len(emotions) < 2:
            return 1.0
        
        # Calculate valence and arousal consistency
        valences = [e.valence for e in emotions]
        arousals = [e.arousal for e in emotions]
        
        valence_std = np.std(valences)
        arousal_std = np.std(arousals)
        
        # Lower std = higher consistency
        # Normalize to 0-1 range
        valence_consistency = 1.0 / (1.0 + valence_std)
        arousal_consistency = 1.0 / (1.0 + arousal_std)
        
        return (valence_consistency + arousal_consistency) / 2
    
    def adjust_semantic_score(
        self,
        base_score: float,
        response_emotion: EmotionProfile,
        context_type: Optional[str] = None,
        weight: float = 0.3
    ) -> float:
        """
        Adjust semantic score based on emotional appropriateness
        
        Args:
            base_score: Base semantic similarity score
            response_emotion: Detected emotion
            context_type: Optional context type
            weight: Weight for emotion adjustment (0-1)
            
        Returns:
            Adjusted score
        """
        if context_type is None:
            # No adjustment if no context
            return base_score
        
        # Get emotional appropriateness
        emotion_score = self.score_emotional_appropriateness(
            response_emotion,
            context_type
        )
        
        # Weighted combination
        adjusted = base_score * (1 - weight) + emotion_score * weight
        
        return adjusted
    
    def get_emotion_feedback(
        self,
        emotion: EmotionProfile,
        context_type: str
    ) -> Dict[str, any]:
        """
        Get feedback on emotional appropriateness
        
        Args:
            emotion: Detected emotion
            context_type: Context type
            
        Returns:
            Feedback dictionary
        """
        appropriateness = self.score_emotional_appropriateness(emotion, context_type)
        appropriate_emotions = self.CONTEXT_EMOTIONS.get(context_type, [])
        
        feedback = {
            'is_appropriate': appropriateness > 0.5,
            'appropriateness_score': appropriateness,
            'detected_emotion': emotion.primary_emotion,
            'confidence': emotion.confidence,
            'expected_emotions': appropriate_emotions,
            'valence': emotion.valence,
            'arousal': emotion.arousal
        }
        
        # Add suggestion if inappropriate
        if appropriateness < 0.5:
            feedback['suggestion'] = f"Consider using emotions: {', '.join(appropriate_emotions)}"
        
        return feedback


# Convenience functions
def analyze_emotion(text: str, use_gpu: bool = True) -> EmotionProfile:
    """Quick emotion analysis"""
    scorer = EmotionAwareScoring(use_gpu=use_gpu)
    return scorer.analyze_emotion(text)


def score_with_emotion(
    base_score: float,
    response: str,
    context_type: str,
    use_gpu: bool = True
) -> Tuple[float, EmotionProfile]:
    """Score with emotion adjustment"""
    scorer = EmotionAwareScoring(use_gpu=use_gpu)
    emotion = scorer.analyze_emotion(response)
    adjusted_score = scorer.adjust_semantic_score(
        base_score,
        emotion,
        context_type
    )
    return adjusted_score, emotion


if __name__ == "__main__":
    # Example usage
    print("Emotion-Aware Scoring - Example")
    print("="*60)
    
    # Example responses with different emotions
    examples = [
        {
            'text': "I'm so happy to help you with your quest!",
            'context': 'quest_offer'
        },
        {
            'text': "Get out of my shop before I call the guards!",
            'context': 'merchant'
        },
        {
            'text': "Welcome traveler, what brings you here?",
            'context': 'greeting'
        }
    ]
    
    print("\nEmotion Analysis Examples:")
    print("(Note: Requires transformers library)")
    
    for ex in examples:
        print(f"\nText: '{ex['text']}'")
        print(f"Context: {ex['context']}")
        # Uncomment when dependencies installed:
        # scorer = EmotionAwareScoring()
        # emotion = scorer.analyze_emotion(ex['text'])
        # feedback = scorer.get_emotion_feedback(emotion, ex['context'])
        # print(f"Emotion: {emotion.primary_emotion} ({emotion.confidence:.2f})")
        # print(f"Appropriate: {feedback['is_appropriate']}")
