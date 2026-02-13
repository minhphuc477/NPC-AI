#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Components/ActorComponent.h" 
#include "NPCConversationSubsystem.generated.h"

// Forward declaration
class UNPCDialogueComponent;

/**
 * Manages multi-agent conversations and spatial audio/text events
 */
UCLASS()
class NPCINFERENCE_API UNPCConversationSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/** Register an NPC to the society */
	UFUNCTION(BlueprintCallable, Category = "NPC Society")
	void RegisterNPC(UNPCDialogueComponent* NPC);

	/** Unregister an NPC */
	UFUNCTION(BlueprintCallable, Category = "NPC Society")
	void UnregisterNPC(UNPCDialogueComponent* NPC);

	/**
	 * Broadcast a message from an NPC to others nearby
	 * @param Sender The component sending the message
	 * @param Message The text content
	 * @param Radius The hearing radius in units (default 1000 = 10m)
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Society")
	void BroadcastMessage(UNPCDialogueComponent* Sender, const FString& Message, float Radius = 1000.0f);

private:
	UPROPERTY()
	TArray<UNPCDialogueComponent*> ActiveNPCs;
};
