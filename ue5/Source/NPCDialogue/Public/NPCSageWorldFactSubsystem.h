// NPCSageWorldFactSubsystem.h
// Shared in-session world fact memory for cross-NPC consistency.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "NPCSageWorldFactSubsystem.generated.h"

USTRUCT(BlueprintType)
struct NPCDIALOGUE_API FNPCSageWorldFactRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString Key;

    UPROPERTY(BlueprintReadOnly)
    FString Value;

    UPROPERTY(BlueprintReadOnly)
    FName SourceNPC;

    UPROPERTY(BlueprintReadOnly)
    FString UpdatedUtc;
};

UCLASS()
class NPCDIALOGUE_API UNPCSageWorldFactSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE|WorldFacts")
    void BroadcastFact(const FString& Key, const FString& Value, FName NPCID);

    UFUNCTION(BlueprintPure, Category = "NPC|SAGE|WorldFacts")
    bool GetFact(const FString& Key, FString& OutValue) const;

    UFUNCTION(BlueprintPure, Category = "NPC|SAGE|WorldFacts")
    bool CheckConflict(const FString& Key, const FString& ProposedValue) const;

    UFUNCTION(BlueprintPure, Category = "NPC|SAGE|WorldFacts")
    TArray<FNPCSageWorldFactRecord> GetAllFacts() const;

    UFUNCTION(BlueprintPure, Category = "NPC|SAGE|WorldFacts")
    FString BuildSummaryForNPC(FName NPCID, int32 MaxFacts = 8) const;

private:
    UPROPERTY()
    TMap<FString, FNPCSageWorldFactRecord> FactsByKey;
};

