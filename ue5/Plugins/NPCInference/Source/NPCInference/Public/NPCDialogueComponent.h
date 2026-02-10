#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NPCDialogueComponent.generated.h"

// Forward declarations
class AAIController;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNPCResponseGenerated, const FString&, Response);

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

	/**
	 * Request a response from the NPC asynchronously
	 * @param PlayerInput The text the player said
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Inference")
	void RequestResponse(const FString& PlayerInput);

	/**
	 * Get current dynamic context as scenario string
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Inference")
	FString GetDynamicScenario();

private:
	/** Extract dynamic context from game state */
	FString ExtractDynamicContext();
};

