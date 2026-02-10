// NPCContextExtractor.h
// Dynamic context extraction from UE5 game state
// Replaces hardcoded context with real-time game information

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "AIController.h"
#include "NPCContextExtractor.generated.h"

/**
 * Extracted context information from UE5
 */
USTRUCT(BlueprintType)
struct FNPCDynamicContext
{
    GENERATED_BODY()

    // Location information
    UPROPERTY(BlueprintReadWrite)
    FString LocationName;

    UPROPERTY(BlueprintReadWrite)
    FVector Position;

    UPROPERTY(BlueprintReadWrite)
    FString ZoneName;

    // Behavior Tree state
    UPROPERTY(BlueprintReadWrite)
    FString CurrentBehavior;

    UPROPERTY(BlueprintReadWrite)
    FString BehaviorState;

    UPROPERTY(BlueprintReadWrite)
    TMap<FString, FString> BlackboardValues;

    // Perception info
    UPROPERTY(BlueprintReadWrite)
    TArray<FString> VisibleActors;

    UPROPERTY(BlueprintReadWrite)
    TArray<FString> HeardSounds;

    UPROPERTY(BlueprintReadWrite)
    bool bCanSeePlayer;

    // Nearby entities
    UPROPERTY(BlueprintReadWrite)
    TArray<FString> NearbyActors;

    UPROPERTY(BlueprintReadWrite)
    TArray<FString> NearbyPlayers;

    UPROPERTY(BlueprintReadWrite)
    float NearestPlayerDistance;

    // Time and environment
    UPROPERTY(BlueprintReadWrite)
    FString TimeOfDay;

    UPROPERTY(BlueprintReadWrite)
    FString Weather;

    // NPC identity
    UPROPERTY(BlueprintReadWrite)
    FString NPCID;

    UPROPERTY(BlueprintReadWrite)
    FString NPCRole;

    // Recent events
    UPROPERTY(BlueprintReadWrite)
    TArray<FString> RecentEvents;
};

/**
 * Extracts dynamic context from UE5 game state
 */
UCLASS()
class UNPCContextExtractor : public UObject
{
    GENERATED_BODY()

public:
    /**
     * Extract complete context from NPC and environment
     */
    UFUNCTION(BlueprintCallable, Category = "NPC|Context")
    static FNPCDynamicContext ExtractContext(
        AActor* NPCActor,
        AAIController* AIController = nullptr,
        float ScanRadius = 1000.0f
    );

    /**
     * Extract location information
     */
    UFUNCTION(BlueprintCallable, Category = "NPC|Context")
    static void ExtractLocationInfo(
        AActor* NPCActor,
        FNPCDynamicContext& OutContext
    );

    /**
     * Extract Behavior Tree state
     */
    UFUNCTION(BlueprintCallable, Category = "NPC|Context")
    static void ExtractBehaviorTreeState(
        AAIController* AIController,
        FNPCDynamicContext& OutContext
    );

    /**
     * Extract nearby entities
     */
    UFUNCTION(BlueprintCallable, Category = "NPC|Context")
    static void ExtractNearbyEntities(
        AActor* NPCActor,
        float ScanRadius,
        FNPCDynamicContext& OutContext
    );

    /**
     * Extract perception info (sight/hearing)
     */
    UFUNCTION(BlueprintCallable, Category = "NPC|Context")
    static void ExtractPerceptionInfo(
        AAIController* AIController,
        FNPCDynamicContext& OutContext
    );

    /**
     * Extract time and weather
     */
    UFUNCTION(BlueprintCallable, Category = "NPC|Context")
    static void ExtractEnvironmentInfo(
        UWorld* World,
        FNPCDynamicContext& OutContext
    );

    /**
     * Format context as scenario string for AI
     */
    UFUNCTION(BlueprintCallable, Category = "NPC|Context")
    static FString FormatContextAsScenario(
        const FNPCDynamicContext& Context
    );

    /**
     * Get blackboard value as string
     */
    static FString GetBlackboardValueAsString(
        UBlackboardComponent* Blackboard,
        const FName& KeyName
    );

    /**
     * Detect zone/location name from position
     */
    static FString DetectZoneName(
        UWorld* World,
        const FVector& Position
    );
};
