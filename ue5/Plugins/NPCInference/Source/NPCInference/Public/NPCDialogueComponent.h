#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NPCDialogueComponent.generated.h"

// Forward declarations
class AAIController;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNPCResponseGenerated, const FString&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNPCSpeechChunkGenerated, const FString&, SpeechChunk);

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class NPCINFERENCE_API UNPCDialogueComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UNPCDialogueComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

public:	
	
	/** Unique ID for this NPC (can be auto-generated from actor name) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile")
	FString NPCID = "";

	/** Persona description */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile", meta = (MultiLine = true))
	FString Persona = "You are a loyal guard.";

	/** 
	 * Scenario/context - DEPRECATED: Use dynamic context extraction instead
	 * Leave empty to auto-extract from game state
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile", meta = (MultiLine = true))
	FString Scenario = "";

	/** Enable dynamic context extraction from UE5 game state */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile")
	bool bUseDynamicContext = true;

	/** Scan radius for nearby entities (in cm) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile")
	float ContextScanRadius = 1000.0f;

	/** Reference to AI Controller (auto-detected if null) */
	UPROPERTY(BlueprintReadWrite, Category = "NPC Profile")
	AAIController* AIController = nullptr;

	/** Event fired when response is generated */
	UPROPERTY(BlueprintAssignable, Category = "NPC Inference")
	FOnNPCResponseGenerated OnResponseGenerated;

	/** Phase 7: Event fired when a chunk of text is generated (for TTS/Lipsync Multimodal hooks) */
	UPROPERTY(BlueprintAssignable, Category = "NPC Inference")
	FOnNPCSpeechChunkGenerated OnSpeechChunkGenerated;

	/** Request a response from the AI engine */
	UFUNCTION(BlueprintCallable, Category = "NPC AI")
	void RequestResponse(const FString& PlayerInput);

	/** Callback for async response received */
	UFUNCTION()
	void OnAsyncResponseReceived(const FString& Response);

	/**
	 * Get current dynamic context as scenario string
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Inference")
	FString GetDynamicScenario();

	/**
	 * NPC hears something interesting
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Gossip")
	void HearGossip(const FString& GossipText, const FString& SourceName);

	/**
	 * NPC shares something interesting
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Gossip")
	FString ShareGossip();

	/**
	 * Trigger sleep/memory consolidation
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Memory")
	void Sleep();

	/**
	 * Called by Subsystem when another NPC speaks nearby
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Society")
	void ReceiveBroadcast(const FString& SenderName, const FString& Message);

	/**
	 * Speak to nearby NPCs
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Society")
	void BroadcastToNearby(const FString& Message);

private:
	/** Extract dynamic context from game state */
	FString ExtractDynamicContext();

	/** Phase 8: Stale Context Prevention state */
	bool bIsGeneratingResponse = false;
	FVector GenerationStartLocation;
	UPROPERTY(Transient)
	AActor* CurrentInteractingPlayer = nullptr;
};

