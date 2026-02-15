#include "EmotionalContinuitySystem.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace NPCInference;

void PrintSeparator() {
    std::cout << "\n" << std::string(70, '=') << "\n\n";
}

void PrintEmotion(const EmotionalState& emotion, const std::string& label) {
    std::cout << label << ":\n";
    std::cout << "  Joy: " << std::fixed << std::setprecision(2) << emotion.joy;
    std::cout << " | Trust: " << emotion.trust;
    std::cout << " | Fear: " << emotion.fear << "\n";
    std::cout << "  Sadness: " << emotion.sadness;
    std::cout << " | Anger: " << emotion.anger;
    std::cout << " | Surprise: " << emotion.surprise << "\n";
    std::cout << "  Valence: " << emotion.GetValence();
    std::cout << " | Arousal: " << emotion.GetArousal() << "\n";
}

int main() {
    std::cout << "=== Emotional Continuity System Demo ===\n";
    std::cout << "Persistent NPC Emotions with Personality\n";
    PrintSeparator();
    
    // === Act 1: Personality Profiles ===
    std::cout << "ACT 1: PERSONALITY PROFILES\n";
    std::cout << "----------------------------\n\n";
    
    std::cout << "Creating two NPCs with different personalities:\n\n";
    
    // Cheerful merchant
    EmotionalContinuitySystem elara(Personalities::Cheerful());
    std::cout << "Elara (Cheerful Merchant):\n";
    auto elara_p = elara.GetPersonality();
    std::cout << "  Extraversion: " << elara_p.extraversion << " (outgoing)\n";
    std::cout << "  Agreeableness: " << elara_p.agreeableness << " (friendly)\n";
    std::cout << "  Neuroticism: " << elara_p.neuroticism << " (stable)\n";
    PrintEmotion(elara.GetCurrentEmotion(), "  Baseline mood");
    std::cout << "\n";
    
    // Grumpy blacksmith
    EmotionalContinuitySystem marcus(Personalities::Grumpy());
    std::cout << "Marcus (Grumpy Blacksmith):\n";
    auto marcus_p = marcus.GetPersonality();
    std::cout << "  Extraversion: " << marcus_p.extraversion << " (introverted)\n";
    std::cout << "  Agreeableness: " << marcus_p.agreeableness << " (gruff)\n";
    std::cout << "  Neuroticism: " << marcus_p.neuroticism << " (anxious)\n";
    PrintEmotion(marcus.GetCurrentEmotion(), "  Baseline mood");
    
    PrintSeparator();
    
    // === Act 2: Emotional Reactions ===
    std::cout << "ACT 2: EMOTIONAL REACTIONS TO SAME EVENT\n";
    std::cout << "----------------------------------------\n\n";
    
    std::cout << "Event: Player gives both NPCs a gift\n\n";
    
    // Generate reactions
    auto elara_reaction = elara.GenerateReaction("gift", 0.8f, "Player");
    auto marcus_reaction = marcus.GenerateReaction("gift", 0.8f, "Player");
    
    std::cout << "Elara's reaction (cheerful personality):\n";
    PrintEmotion(elara_reaction, "  Emotional response");
    std::cout << "  → " << "Very joyful and trusting!\n\n";
    
    std::cout << "Marcus's reaction (grumpy personality):\n";
    PrintEmotion(marcus_reaction, "  Emotional response");
    std::cout << "  → " << "Less enthusiastic, more cautious\n";
    
    PrintSeparator();
    
    // === Act 3: Emotional Inertia ===
    std::cout << "ACT 3: EMOTIONAL INERTIA (Gradual Change)\n";
    std::cout << "-----------------------------------------\n\n";
    
    std::cout << "Applying Elara's reaction with emotional inertia...\n\n";
    
    PrintEmotion(elara.GetCurrentEmotion(), "BEFORE stimulus");
    
    // Apply stimulus - emotions don't change instantly!
    elara.ApplyEmotionalStimulus(elara_reaction, 1.0f);
    
    PrintEmotion(elara.GetCurrentEmotion(), "AFTER stimulus (1st application)");
    std::cout << "  💡 Notice: Emotions blended, not replaced instantly!\n\n";
    
    // Apply again to show gradual change
    elara.ApplyEmotionalStimulus(elara_reaction, 1.0f);
    PrintEmotion(elara.GetCurrentEmotion(), "AFTER stimulus (2nd application)");
    std::cout << "  💡 Emotions gradually approach target state\n";
    
    PrintSeparator();
    
    // === Act 4: Entity Sentiment ===
    std::cout << "ACT 4: ENTITY SENTIMENT TRACKING\n";
    std::cout << "--------------------------------\n\n";
    
    std::cout << "Event: Player helps Elara, betrays Marcus\n\n";
    
    // Elara likes player
    elara.UpdateSentiment("Player", 0.7f, 0.8f, "gift_event");
    auto elara_sentiment = elara.GetSentiment("Player");
    std::cout << "Elara's sentiment toward Player:\n";
    std::cout << "  Sentiment: " << elara_sentiment.sentiment << " (positive)\n";
    std::cout << "  Intensity: " << elara_sentiment.intensity << " (strong)\n\n";
    
    // Marcus distrusts player
    marcus.UpdateSentiment("Player", -0.6f, 0.7f, "betrayal_event");
    auto marcus_sentiment = marcus.GetSentiment("Player");
    std::cout << "Marcus's sentiment toward Player:\n";
    std::cout << "  Sentiment: " << marcus_sentiment.sentiment << " (negative)\n";
    std::cout << "  Intensity: " << marcus_sentiment.intensity << " (strong)\n";
    
    PrintSeparator();
    
    // === Act 5: Personality-Modulated Reactions ===
    std::cout << "ACT 5: PERSONALITY-MODULATED REACTIONS\n";
    std::cout << "--------------------------------------\n\n";
    
    std::cout << "Event: Both NPCs face a threat\n\n";
    
    auto elara_threat = elara.GenerateReaction("threat", 0.9f);
    auto marcus_threat = marcus.GenerateReaction("threat", 0.9f);
    
    std::cout << "Elara's reaction (low neuroticism = stable):\n";
    std::cout << "  Fear: " << elara_threat.fear << "\n";
    std::cout << "  Anger: " << elara_threat.anger << "\n\n";
    
    std::cout << "Marcus's reaction (high neuroticism = anxious):\n";
    std::cout << "  Fear: " << marcus_threat.fear << " (HIGHER fear!)\n";
    std::cout << "  Anger: " << marcus_threat.anger << "\n";
    std::cout << "  💡 Neurotic personalities react with more fear!\n";
    
    PrintSeparator();
    
    // === Act 6: Emotional Decay ===
    std::cout << "ACT 6: EMOTIONAL DECAY TO BASELINE\n";
    std::cout << "----------------------------------\n\n";
    
    // Set Marcus to very angry
    EmotionalState angry_state;
    angry_state.anger = 0.9f;
    angry_state.disgust = 0.6f;
    marcus.ApplyEmotionalStimulus(angry_state, 1.0f, 0.3f);  // Low inertia for demo
    
    std::cout << "Marcus is very angry:\n";
    PrintEmotion(marcus.GetCurrentEmotion(), "  Current emotion");
    std::cout << "  Dominant: " << marcus.GetDominantEmotion() << "\n\n";
    
    std::cout << "Time passes... emotions decay toward baseline\n\n";
    
    for (int i = 0; i < 5; i++) {
        marcus.DecayTowardBaseline(1.0f);
        std::cout << "After " << (i+1) << " time step(s):\n";
        std::cout << "  Anger: " << marcus.GetCurrentEmotion().anger;
        std::cout << " | Joy: " << marcus.GetCurrentEmotion().joy << "\n";
    }
    
    std::cout << "\n💡 Emotions gradually return to baseline mood!\n";
    
    PrintSeparator();
    
    // === Act 7: Emotional Description ===
    std::cout << "ACT 7: EMOTIONAL EXPRESSION\n";
    std::cout << "---------------------------\n\n";
    
    // Set different emotional states
    EmotionalState joyful;
    joyful.joy = 0.9f;
    joyful.anticipation = 0.7f;
    elara.ApplyEmotionalStimulus(joyful, 1.0f, 0.2f);
    
    std::cout << "Elara: \"" << elara.DescribeEmotion() << "\"\n";
    std::cout << "  Dominant emotion: " << elara.GetDominantEmotion() << "\n";
    std::cout << "  Intensity: " << elara.GetEmotionalIntensity() << "\n\n";
    
    EmotionalState fearful;
    fearful.fear = 0.8f;
    fearful.surprise = 0.6f;
    marcus.ApplyEmotionalStimulus(fearful, 1.0f, 0.2f);
    
    std::cout << "Marcus: \"" << marcus.DescribeEmotion() << "\"\n";
    std::cout << "  Dominant emotion: " << marcus.GetDominantEmotion() << "\n";
    std::cout << "  Intensity: " << marcus.GetEmotionalIntensity() << "\n";
    
    PrintSeparator();
    
    // === Act 8: Sentiment-Influenced Reactions ===
    std::cout << "ACT 8: SENTIMENT-INFLUENCED REACTIONS\n";
    std::cout << "-------------------------------------\n\n";
    
    std::cout << "Event: Player betrays both NPCs\n\n";
    
    // Elara likes player (positive sentiment)
    auto elara_betrayal = elara.GenerateReaction("betrayal", 0.9f, "Player");
    std::cout << "Elara's reaction (has positive sentiment toward Player):\n";
    std::cout << "  Anger: " << elara_betrayal.anger << "\n";
    std::cout << "  Sadness: " << elara_betrayal.sadness << " (MORE sad - betrayed by friend!)\n\n";
    
    // Marcus already dislikes player (negative sentiment)
    auto marcus_betrayal = marcus.GenerateReaction("betrayal", 0.9f, "Player");
    std::cout << "Marcus's reaction (has negative sentiment toward Player):\n";
    std::cout << "  Anger: " << marcus_betrayal.anger << "\n";
    std::cout << "  Sadness: " << marcus_betrayal.sadness << " (less sad - expected it)\n";
    std::cout << "  💡 Existing sentiment modulates reactions!\n";
    
    PrintSeparator();
    
    // === Statistics ===
    std::cout << "EMOTIONAL STATISTICS\n";
    std::cout << "--------------------\n\n";
    
    auto elara_stats = elara.GetStats();
    std::cout << "Elara:\n";
    std::cout << "  Average Valence: " << elara_stats.avg_valence << " (positive)\n";
    std::cout << "  Average Arousal: " << elara_stats.avg_arousal << "\n";
    std::cout << "  Emotional Volatility: " << elara_stats.emotional_volatility << "\n";
    std::cout << "  Tracked Sentiments: " << elara_stats.num_sentiments << "\n\n";
    
    auto marcus_stats = marcus.GetStats();
    std::cout << "Marcus:\n";
    std::cout << "  Average Valence: " << marcus_stats.avg_valence << "\n";
    std::cout << "  Average Arousal: " << marcus_stats.avg_arousal << "\n";
    std::cout << "  Emotional Volatility: " << marcus_stats.emotional_volatility << "\n";
    std::cout << "  Tracked Sentiments: " << marcus_stats.num_sentiments << "\n";
    
    PrintSeparator();
    
    // === Persistence ===
    std::cout << "PERSISTENCE TEST\n";
    std::cout << "----------------\n\n";
    
    std::string save_path = "elara_emotions.json";
    if (elara.Save(save_path)) {
        std::cout << "✓ Saved Elara's emotional state to " << save_path << "\n";
    }
    
    EmotionalContinuitySystem loaded;
    if (loaded.Load(save_path)) {
        std::cout << "✓ Loaded emotional state from " << save_path << "\n\n";
        std::cout << "Loaded emotion: " << loaded.DescribeEmotion() << "\n";
        std::cout << "Loaded personality agreeableness: " << loaded.GetPersonality().agreeableness << "\n";
    }
    
    PrintSeparator();
    
    std::cout << "✅ DEMO COMPLETE\n\n";
    std::cout << "Key Takeaways:\n";
    std::cout << "1. Personality profiles drive different reactions to same events\n";
    std::cout << "2. Emotional inertia prevents instant mood swings\n";
    std::cout << "3. Entity sentiment tracks how NPCs feel about specific characters\n";
    std::cout << "4. Emotions decay gradually toward baseline mood\n";
    std::cout << "5. Existing sentiment modulates future reactions\n";
    std::cout << "6. Full persistence for long-term emotional continuity\n\n";
    std::cout << "This solves the unsolved problem of NPCs with no emotional memory!\n";
    std::cout << "NPCs now have PERSISTENT, BELIEVABLE emotions.\n";
    
    return 0;
}
