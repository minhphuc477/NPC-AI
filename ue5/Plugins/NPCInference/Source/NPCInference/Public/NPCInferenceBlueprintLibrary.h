#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "NPCInferenceBlueprintLibrary.generated.h"

/**
 * Helper functions for NPC Inference
 */
UCLASS()
class NPCINFERENCE_API UNPCInferenceBlueprintLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Check if the NPC Inference Engine is available and ready
	 */
	UFUNCTION(BlueprintPure, Category = "NPC Inference", meta = (WorldContext = "WorldContextObject"))
	static bool IsNPCEngineReady(const UObject* WorldContextObject);

	/**
	 * Quick synchronous generation (Warning: Blocks Game Thread!)
	 * Only use for testing or very short responses
	 */
	UFUNCTION(BlueprintCallable, Category = "NPC Inference", meta = (WorldContext = "WorldContextObject"))
	static FString GenerateDialogueImmediate(const UObject* WorldContextObject, FString System, FString Name, FString Context, FString Input);

    /**
     * Get the suggested model path relative to the project content
     */
    UFUNCTION(BlueprintPure, Category = "NPC Inference")
    static FString GetDefaultModelPath();

    /**
     * Run an automated Benchmark from a JSON file to a JSONL log file.
     * Operates purely on background threads to prevent UE5 Game Thread freezing.
     */
    UFUNCTION(BlueprintCallable, Category = "NPC Inference|Benchmark", meta = (WorldContext = "WorldContextObject"))
    static void RunMockBenchmark(const UObject* WorldContextObject, FString InputJsonPath, FString OutputLogPath);
};
