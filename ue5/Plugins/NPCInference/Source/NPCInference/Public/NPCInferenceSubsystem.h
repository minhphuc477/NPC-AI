#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "NPCInference.h" // C++ Engine Header
#include "NPCInferenceSubsystem.generated.h"

/**
 * Subsystem to manage the lifecycle of the NPC Inference Engine
 */
UCLASS()
class NPCINFERENCE_API UNPCInferenceSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	// Begin USubsystem
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	// End USubsystem

	/**
	 * Initialize the engine with a specific model path
	 * @param ModelPath Absolute path to the ONNX model directory or tokenizer file location
	 * @return true if successful
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Inference")
	bool InitializeEngine(const FString& ModelPath);

	/**
	 * Generate a response for an NPC
	 * @param SystemPrompt The persona/system instruction
	 * @param Name NPC Name
	 * @param Context The scenario or context
	 * @param PlayerInput What the player said
	 * @return The generated response
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Inference")
	FString GenerateDialogue(const FString& SystemPrompt, const FString& Name, const FString& Context, const FString& PlayerInput);

	/**
	 * Native generation wrapper
	 */
	FString GenerateFromPrompt(const FString& FullPrompt);

	/** Check if engine is ready */
	UFUNCTION(BlueprintPure, Category = "NPC Inference")
	bool IsEngineReady() const;

private:
	// Pimpl idiom or direct member if header is available
	// Since we include NPCInference.h, we can use the class directly or via pointer
	// Using pointer to avoid strict dependency in header if possible, but here we included it.
	std::unique_ptr<NPCInference::NPCInferenceEngine> InferenceEngine;

	// Helper to convert FString to std::string
	std::string ToString(const FString& InStr);
	// Helper to convert std::string to FString
	FString ToFString(const std::string& InStr);
};
