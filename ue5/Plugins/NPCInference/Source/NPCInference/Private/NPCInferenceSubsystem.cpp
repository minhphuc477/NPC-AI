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

FString UNPCInferenceSubsystem::GenerateStructuredDialogue(const FString& SystemPrompt, const FString& Name, const FString& Context, const FString& PlayerInput)
{
    if (!IsEngineReady()) return TEXT("Error: Engine not ready");
    
    // Construct prompt manually or use a helper
    std::string prompt = "System: " + ToString(SystemPrompt) + "\nName: " + ToString(Name) + "\nContext: " + ToString(Context) + "\n\nQuestion: " + ToString(PlayerInput) + "\nAnswer:";
    
    std::string response = InferenceEngine->GenerateJSON(prompt);
    return ToFString(response);
}

FString UNPCInferenceSubsystem::ExecuteNPCTool(const FString& ToolCallJSON)
{
    if (!InferenceEngine) return TEXT("Error: Engine not ready");
    return ToFString(InferenceEngine->ExecuteAction(ToString(ToolCallJSON)));
}

void UNPCInferenceSubsystem::GenerateDialogueAsync(const FString& SystemPrompt, const FString& Name, const FString& Context, const FString& PlayerInput, FOnDialogueGenerated OnComplete)
{
    if (!IsEngineReady())
    {
        OnComplete.ExecuteIfBound(TEXT("Error: Engine not ready"));
        return;
    }

    // Use UE5 AsyncTask to run generation on a background thread
    AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, SystemPrompt, Name, Context, PlayerInput, OnComplete]()
    {
        std::string StdResponse = InferenceEngine->GenerateFromContext(
            ToString(SystemPrompt),
            ToString(Name),
            ToString(Context),
            ToString(PlayerInput)
        );

        FString FinalResponse = ToFString(StdResponse);

        // Return to Game Thread to execute the delegate
        AsyncTask(ENamedThreads::GameThread, [OnComplete, FinalResponse]()
        {
            OnComplete.ExecuteIfBound(FinalResponse);
        });
    });
}

void UNPCInferenceSubsystem::ReceiveGossip(const FString& GossipText, const FString& SourceNPC)
{
    if (InferenceEngine)
    {
        InferenceEngine->ReceiveGossip(ToString(GossipText), ToString(SourceNPC));
    }
}

FString UNPCInferenceSubsystem::ExtractGossip()
{
    if (InferenceEngine)
    {
        return ToFString(InferenceEngine->ExtractGossip());
    }
    return TEXT("");
}

void UNPCInferenceSubsystem::TriggerSleepMode()
{
    if (InferenceEngine)
    {
        InferenceEngine->PerformSleepCycle();
    }
}

FString UNPCInferenceSubsystem::AnalyzeScene(const TArray<uint8>& ImageData, int32 Width, int32 Height)
{
    if (!InferenceEngine) return TEXT("Error: Engine not ready");

    // Convert TArray to std::vector
    std::vector<uint8_t> std_data(ImageData.GetData(), ImageData.GetData() + ImageData.Num());
    
    std::string desc = InferenceEngine->See(std_data, Width, Height);
    return ToFString(desc);
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
