// NPCSageBlackboardKeys.h
// Concrete Blackboard key map for SAGE BT assets.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "NPCSageBlackboardKeys.generated.h"

namespace NPCSageBlackboardKeys
{
    extern const FName NPCId;
    extern const FName PlayerId;
    extern const FName SessionId;

    extern const FName BehaviorState;
    extern const FName Location;
    extern const FName Persona;
    extern const FName PlayerQuery;
    extern const FName PlayerInput;
    extern const FName CandidateResponse;
    extern const FName NPCResponse;

    extern const FName ThreatEventQueue;
    extern const FName NearbyThreat;
    extern const FName IsInCombat;
    extern const FName InterruptFlag;
    extern const FName PlayerDistance;
    extern const FName NPCHealth;

    extern const FName ActiveQuestPhase;
    extern const FName QuestPhaseSource;

    extern const FName SessionInitDone;
    extern const FName StateTransitionFlag;
    extern const FName PrefixCacheValid;
    extern const FName GameStateJson;
    extern const FName PrefixInvalidationResult;
    extern const FName CurrentStateSnapshot;
    extern const FName LastStateSnapshot;
    extern const FName StateHash;
    extern const FName SessionTurnCount;
    extern const FName MoodState;
    extern const FName TrustScore;
    extern const FName RelationshipScore;
    extern const FName TrustEvent;
    extern const FName GenerationTTFT;
    extern const FName FallbackUsed;

    extern const FName EpisodicMemoryHandle;
    extern const FName EpisodicContext;
    extern const FName EpisodicMemoryFormatted;
    extern const FName ExtractResult;

    extern const FName WorldFacts;
    extern const FName PrefetchedPassages;
    extern const FName PrefetchResult;
    extern const FName ConsistencyViolation;

    extern const FName ImplicitFeedbackScore;
    extern const FName FeedbackOutcome;
    extern const FName FeedbackLogResult;
}

USTRUCT(BlueprintType)
struct NPCDIALOGUE_API FNPCSageBlackboardKeySpec
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FName KeyName;

    UPROPERTY(BlueprintReadOnly)
    FString ValueType;

    UPROPERTY(BlueprintReadOnly)
    FString Purpose;
};

UCLASS()
class NPCDIALOGUE_API UNPCSageBlackboardKeyLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintPure, Category = "NPC|SAGE")
    static TArray<FNPCSageBlackboardKeySpec> GetDefaultSageBlackboardSchema();
};
