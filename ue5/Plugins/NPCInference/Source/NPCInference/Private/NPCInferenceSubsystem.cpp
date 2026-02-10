#include "NPCInferenceSubsystem.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"

void UNPCInferenceSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	InferenceEngine = std::make_unique<NPCInference::NPCInferenceEngine>();
	
	// Try to auto-locate model in Game/Content/ModelData if possible, or wait for explicit Initialize
	// For now, we wait for explicit BP call or check a config
}

void UNPCInferenceSubsystem::Deinitialize()
{
	InferenceEngine.reset();
	Super::Deinitialize();
}

bool UNPCInferenceSubsystem::InitializeEngine(const FString& ModelPath)
{
	if (!InferenceEngine)
	{
		InferenceEngine = std::make_unique<NPCInference::NPCInferenceEngine>();
	}

	std::string StdModelPath = ToString(ModelPath);
	
	// If path is relative, make it absolute relative to Project Content?
	// The C++ engine expects a path where it can find tokenizer.model etc.
	
	UE_LOG(LogTemp, Log, TEXT("NPCInference: Initializing with path: %s"), *ModelPath);
	
	bool bSuccess = InferenceEngine->Initialize(StdModelPath);
	if (bSuccess)
	{
		UE_LOG(LogTemp, Log, TEXT("NPCInference: Engine Initialized Successfully!"));
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("NPCInference: Failed to initialize engine."));
	}
	
	return bSuccess;
}

FString UNPCInferenceSubsystem::GenerateDialogue(const FString& SystemPrompt, const FString& Name, const FString& Context, const FString& PlayerInput)
{
	if (!IsEngineReady())
	{
		UE_LOG(LogTemp, Warning, TEXT("NPCInference: Engine not ready."));
		return FString("Error: Engine not ready");
	}

	std::string StdResponse = InferenceEngine->GenerateFromContext(
		ToString(SystemPrompt),
		ToString(Name),
		ToString(Context),
		ToString(PlayerInput)
	);

	return ToFString(StdResponse);
}

FString UNPCInferenceSubsystem::GenerateFromPrompt(const FString& FullPrompt)
{
	if (!IsEngineReady()) return "Error: Engine not ready";
	return ToFString(InferenceEngine->Generate(ToString(FullPrompt)));
}

bool UNPCInferenceSubsystem::IsEngineReady() const
{
	return InferenceEngine && InferenceEngine->IsReady();
}

std::string UNPCInferenceSubsystem::ToString(const FString& InStr)
{
	return std::string(TCHAR_TO_UTF8(*InStr));
}

FString UNPCInferenceSubsystem::ToFString(const std::string& InStr)
{
	return FString(UTF8_TO_TCHAR(InStr.c_str()));
}
