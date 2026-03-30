// NPCSageBlackboardKeys.cpp

#include "NPCSageBlackboardKeys.h"

namespace NPCSageBlackboardKeys
{
    const FName NPCId(TEXT("NPCId"));
    const FName PlayerId(TEXT("PlayerId"));
    const FName SessionId(TEXT("SessionId"));

    const FName BehaviorState(TEXT("BehaviorState"));
    const FName Location(TEXT("Location"));
    const FName Persona(TEXT("Persona"));
    const FName PlayerQuery(TEXT("PlayerQuery"));
    const FName PlayerInput(TEXT("PlayerInput"));
    const FName CandidateResponse(TEXT("CandidateResponse"));
    const FName NPCResponse(TEXT("NPCResponse"));

    const FName ThreatEventQueue(TEXT("ThreatEventQueue"));
    const FName NearbyThreat(TEXT("NearbyThreat"));
    const FName IsInCombat(TEXT("IsInCombat"));
    const FName InterruptFlag(TEXT("InterruptFlag"));
    const FName PlayerDistance(TEXT("PlayerDistance"));
    const FName NPCHealth(TEXT("NPCHealth"));

    const FName ActiveQuestPhase(TEXT("ActiveQuestPhase"));
    const FName QuestPhaseSource(TEXT("QuestPhaseSource"));

    const FName SessionInitDone(TEXT("SessionInitDone"));
    const FName StateTransitionFlag(TEXT("StateTransitionFlag"));
    const FName PrefixCacheValid(TEXT("PrefixCacheValid"));
    const FName GameStateJson(TEXT("GameStateJson"));
    const FName PrefixInvalidationResult(TEXT("PrefixInvalidationResult"));
    const FName CurrentStateSnapshot(TEXT("CurrentStateSnapshot"));
    const FName LastStateSnapshot(TEXT("LastStateSnapshot"));
    const FName StateHash(TEXT("StateHash"));
    const FName SessionTurnCount(TEXT("SessionTurnCount"));
    const FName MoodState(TEXT("MoodState"));
    const FName TrustScore(TEXT("TrustScore"));
    const FName RelationshipScore(TEXT("RelationshipScore"));
    const FName TrustEvent(TEXT("TrustEvent"));
    const FName GenerationTTFT(TEXT("GenerationTTFT"));
    const FName FallbackUsed(TEXT("FallbackUsed"));

    const FName EpisodicMemoryHandle(TEXT("EpisodicMemoryHandle"));
    const FName EpisodicContext(TEXT("EpisodicContext"));
    const FName EpisodicMemoryFormatted(TEXT("EpisodicMemoryFormatted"));
    const FName ExtractResult(TEXT("ExtractResult"));

    const FName WorldFacts(TEXT("WorldFacts"));
    const FName PrefetchedPassages(TEXT("PrefetchedPassages"));
    const FName PrefetchResult(TEXT("PrefetchResult"));
    const FName ConsistencyViolation(TEXT("ConsistencyViolation"));

    const FName ImplicitFeedbackScore(TEXT("ImplicitFeedbackScore"));
    const FName FeedbackOutcome(TEXT("FeedbackOutcome"));
    const FName FeedbackLogResult(TEXT("FeedbackLogResult"));
}

TArray<FNPCSageBlackboardKeySpec> UNPCSageBlackboardKeyLibrary::GetDefaultSageBlackboardSchema()
{
    TArray<FNPCSageBlackboardKeySpec> Out;
    auto Add = [&Out](FName Key, const TCHAR* Type, const TCHAR* Purpose)
    {
        FNPCSageBlackboardKeySpec Row;
        Row.KeyName = Key;
        Row.ValueType = FString(Type);
        Row.Purpose = FString(Purpose);
        Out.Add(Row);
    };

    Add(NPCSageBlackboardKeys::NPCId, TEXT("String"), TEXT("Current NPC stable identifier"));
    Add(NPCSageBlackboardKeys::PlayerId, TEXT("String"), TEXT("Current player identifier"));
    Add(NPCSageBlackboardKeys::SessionId, TEXT("String"), TEXT("Dialogue/session identifier"));

    Add(NPCSageBlackboardKeys::BehaviorState, TEXT("String"), TEXT("Current behavior state used by GSPE/SRLE"));
    Add(NPCSageBlackboardKeys::Location, TEXT("String"), TEXT("Current location context"));
    Add(NPCSageBlackboardKeys::Persona, TEXT("String"), TEXT("Persona descriptor/name"));
    Add(NPCSageBlackboardKeys::PlayerQuery, TEXT("String"), TEXT("Current player query used for retrieval"));
    Add(NPCSageBlackboardKeys::PlayerInput, TEXT("String"), TEXT("Raw latest player utterance"));
    Add(NPCSageBlackboardKeys::CandidateResponse, TEXT("String"), TEXT("Generated response under evaluation"));
    Add(NPCSageBlackboardKeys::NPCResponse, TEXT("String"), TEXT("Committed final NPC response"));

    Add(NPCSageBlackboardKeys::ThreatEventQueue, TEXT("String"), TEXT("Threat event queue, serialized"));
    Add(NPCSageBlackboardKeys::NearbyThreat, TEXT("Bool"), TEXT("Fast threat bit from perception/combat"));
    Add(NPCSageBlackboardKeys::IsInCombat, TEXT("Bool"), TEXT("Combat flag from AI state"));
    Add(NPCSageBlackboardKeys::InterruptFlag, TEXT("Bool"), TEXT("Raised by interrupt decorator when dialogue should abort"));
    Add(NPCSageBlackboardKeys::PlayerDistance, TEXT("Float"), TEXT("Distance between NPC and player for interrupt policy"));
    Add(NPCSageBlackboardKeys::NPCHealth, TEXT("Float"), TEXT("Normalized NPC health used by interrupt policy"));

    Add(NPCSageBlackboardKeys::ActiveQuestPhase, TEXT("String"), TEXT("Quest progression phase for routing"));
    Add(NPCSageBlackboardKeys::QuestPhaseSource, TEXT("String"), TEXT("External quest subsystem phase feed"));

    Add(NPCSageBlackboardKeys::SessionInitDone, TEXT("Bool"), TEXT("True once session-init service has hydrated memory/facts"));
    Add(NPCSageBlackboardKeys::StateTransitionFlag, TEXT("Bool"), TEXT("Set true when watched state changes"));
    Add(NPCSageBlackboardKeys::PrefixCacheValid, TEXT("Bool"), TEXT("True when cached GSPE prefix is current"));
    Add(NPCSageBlackboardKeys::GameStateJson, TEXT("String"), TEXT("Serialized game state payload for Python handler"));
    Add(NPCSageBlackboardKeys::PrefixInvalidationResult, TEXT("String"), TEXT("Raw invalidate-prefix-cache handler JSON"));
    Add(NPCSageBlackboardKeys::CurrentStateSnapshot, TEXT("String"), TEXT("Current watched-state snapshot"));
    Add(NPCSageBlackboardKeys::LastStateSnapshot, TEXT("String"), TEXT("Previous watched-state snapshot"));
    Add(NPCSageBlackboardKeys::StateHash, TEXT("String"), TEXT("Deterministic hash over GSPE-driving state fields"));
    Add(NPCSageBlackboardKeys::SessionTurnCount, TEXT("Int"), TEXT("Turn index bucket source for state-conditioned control"));
    Add(NPCSageBlackboardKeys::MoodState, TEXT("String"), TEXT("Calm/tense/agitated/distressed mood signal"));
    Add(NPCSageBlackboardKeys::TrustScore, TEXT("Float"), TEXT("Persistent trust score in [-1,1]"));
    Add(NPCSageBlackboardKeys::RelationshipScore, TEXT("String"), TEXT("Bucketed relationship label from trust score"));
    Add(NPCSageBlackboardKeys::TrustEvent, TEXT("String"), TEXT("Gameplay trust update event token"));
    Add(NPCSageBlackboardKeys::GenerationTTFT, TEXT("Float"), TEXT("Last generation TTFT in milliseconds"));
    Add(NPCSageBlackboardKeys::FallbackUsed, TEXT("Bool"), TEXT("True when fallback response path was used"));

    Add(NPCSageBlackboardKeys::EpisodicMemoryHandle, TEXT("String"), TEXT("Raw load-episodic handler JSON"));
    Add(NPCSageBlackboardKeys::EpisodicContext, TEXT("String"), TEXT("Structured episodic context blob for first-turn hydration"));
    Add(NPCSageBlackboardKeys::EpisodicMemoryFormatted, TEXT("String"), TEXT("Prompt-ready episodic memory snippet"));
    Add(NPCSageBlackboardKeys::ExtractResult, TEXT("String"), TEXT("Raw extract-episodic handler JSON"));

    Add(NPCSageBlackboardKeys::WorldFacts, TEXT("String"), TEXT("Semicolon-delimited world fact summary for consistency guard"));
    Add(NPCSageBlackboardKeys::PrefetchedPassages, TEXT("String"), TEXT("Prefetch payload with predicted next-state evidence"));
    Add(NPCSageBlackboardKeys::PrefetchResult, TEXT("String"), TEXT("Raw prefetch handler JSON"));
    Add(NPCSageBlackboardKeys::ConsistencyViolation, TEXT("Bool"), TEXT("Decorator output; true if contradiction detected"));

    Add(NPCSageBlackboardKeys::ImplicitFeedbackScore, TEXT("Float"), TEXT("Turn-level implicit feedback value"));
    Add(NPCSageBlackboardKeys::FeedbackOutcome, TEXT("String"), TEXT("Feedback label (continued/abandoned/etc.)"));
    Add(NPCSageBlackboardKeys::FeedbackLogResult, TEXT("String"), TEXT("Raw log-feedback handler JSON"));
    return Out;
}
