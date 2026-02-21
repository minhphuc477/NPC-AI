// AsyncGenerateDialogue.h - Async Blueprint Task Node for NPC Dialogue Generation

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintAsyncActionBase.h"
#include "AsyncGenerateDialogue.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnDialogueComplete, const FString&, Response);

/**
 * Async Blueprint node for generating NPC dialogue without blocking the game thread
 * Usage in Blueprint: Async Generate Dialogue -> On Complete / On Failed
 */
UCLASS()
class NPCINFERENCE_API UAsyncGenerateDialogue : public UBlueprintAsyncActionBase
{
	GENERATED_BODY()

public:
	/** Success pin */
	UPROPERTY(BlueprintAssignable)
	FOnDialogueComplete OnComplete;

	/** Failure pin */
	UPROPERTY(BlueprintAssignable)
	FOnDialogueComplete OnFailed;

	/**
	 * Generate NPC dialogue asynchronously
	 * @param WorldContextObject World context
	 * @param SystemPrompt NPC persona/system instruction
	 * @param NPCName Name of the NPC
	 * @param Context Scenario or dynamic context
	 * @param PlayerInput What the player said
	 * @return Async action proxy
	 */
	UFUNCTION(BlueprintCallable, meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject"), Category = "NPC Inference|Async")
	static UAsyncGenerateDialogue* AsyncGenerateDialogue(
		UObject* WorldContextObject,
		const FString& SystemPrompt,
		const FString& NPCName,
		const FString& Context,
		const FString& PlayerInput
	);

	// UBlueprintAsyncActionBase interface
	virtual void Activate() override;

private:
	UObject* WorldContext;
	FString Prompt_System;
	FString Prompt_Name;
	FString Prompt_Context;
	FString Prompt_Input;

	void ExecuteGeneration(TWeakObjectPtr<class UNPCInferenceSubsystem> WeakSubsystem);
	void OnGenerationComplete(const FString& Response);
};
