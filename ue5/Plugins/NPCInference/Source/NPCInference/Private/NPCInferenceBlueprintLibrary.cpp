#include "NPCInferenceBlueprintLibrary.h"
#include "NPCInferenceSubsystem.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/Paths.h"

bool UNPCInferenceBlueprintLibrary::IsNPCEngineReady(const UObject* WorldContextObject)
{
	if (!WorldContextObject) return false;
	
	UGameInstance* GI = UGameplayStatics::GetGameInstance(WorldContextObject);
	if (!GI) return false;

	UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>();
	return Subsystem && Subsystem->IsEngineReady();
}

FString UNPCInferenceBlueprintLibrary::GenerateDialogueImmediate(const UObject* WorldContextObject, FString System, FString Name, FString Context, FString Input)
{
	if (!WorldContextObject) return "Error: Invalid Context";

	UGameInstance* GI = UGameplayStatics::GetGameInstance(WorldContextObject);
	if (!GI) return "Error: No GameInstance";

	UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>();
	if (!Subsystem || !Subsystem->IsEngineReady()) return "Error: Engine Not Ready";

	// Note: You should verify GenerateDialogue uses standard strings or FString correctly in Subsystem
	return Subsystem->GenerateDialogue(System, Name, Context, Input);
}

FString UNPCInferenceBlueprintLibrary::GetDefaultModelPath()
{
    // Return a path relative to the project content
    return FPaths::Combine(FPaths::ProjectContentDir(), TEXT("ModelData"));
}
