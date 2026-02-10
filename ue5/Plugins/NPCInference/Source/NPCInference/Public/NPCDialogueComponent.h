#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NPCDialogueComponent.generated.h"

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
	
	/** Unique ID for this NPC */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile")
	FString NPCID = "Guard_1";

	/** Persona description */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile", meta = (MultiLine = true))
	FString Persona = "You are a loyal guard.";

	/** Current scenario or context */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NPC Profile", meta = (MultiLine = true))
	FString Scenario = "At the village gate.";

	/** Event fired when response is generated */
	UPROPERTY(BlueprintAssignable, Category = "NPC Inference")
	FOnNPCResponseGenerated OnResponseGenerated;

	/**
	 * Request a response from the NPC asynchronously
	 * @param PlayerInput The text the player said
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Inference")
	void RequestResponse(const FString& PlayerInput);
};
