#include "NPCInferenceBlueprintLibrary.h"
#include "NPCInferenceSubsystem.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Async/Async.h"

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

void UNPCInferenceBlueprintLibrary::RunMockBenchmark(const UObject* WorldContextObject, FString InputJsonPath, FString OutputLogPath)
{
    if (!WorldContextObject) return;

    UGameInstance* GI = UGameplayStatics::GetGameInstance(WorldContextObject);
    if (!GI) return;

    UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>();
    if (!Subsystem || !Subsystem->IsEngineReady())
    {
        UE_LOG(LogTemp, Error, TEXT("RunMockBenchmark: C++ Inference Engine Not Ready! Initialize it first."));
        return;
    }

    FString JsonContent;
    // 1. Read test_state.json
    if (!FFileHelper::LoadFileToString(JsonContent, *InputJsonPath)) {
        UE_LOG(LogTemp, Error, TEXT("RunMockBenchmark: Tệp input không tồn tại: %s"), *InputJsonPath);
        return;
    }

    // Capture weak pointers to avoid Use-After-Free crashes if the user closes the game mid-benchmark
    TWeakObjectPtr<UGameInstance> WeakGI(GI);
    TWeakObjectPtr<UNPCInferenceSubsystem> WeakSubsystem(Subsystem);

    // 2. Launch Background Thread Activity (Prevent Frame Freeze)
    AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [JsonContent, OutputLogPath, WeakGI, WeakSubsystem]()
    {
        // 3. Parse JSON Scenarios
        TSharedPtr<FJsonObject> JsonObject;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonContent);
        if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid()) {
            UE_LOG(LogTemp, Error, TEXT("RunMockBenchmark: Lỗi parse JSON structure"));
            return;
        }

        const TArray<TSharedPtr<FJsonValue>>* TestCases;
        if (JsonObject->TryGetArrayField(TEXT("scenarios"), TestCases))
        {
            FString AllLogs = TEXT("");

            for (const TSharedPtr<FJsonValue>& Value : *TestCases)
            {
                // Safety checkpoint: Did they stop PIE (Play in Editor)?
                if (!WeakGI.IsValid() || !WeakSubsystem.IsValid())
                {
                    UE_LOG(LogTemp, Warning, TEXT("RunMockBenchmark: Game Closed, aborting async loop."));
                    return; 
                }

                TSharedPtr<FJsonObject> ScenarioObj = Value->AsObject();
                if (!ScenarioObj.IsValid()) continue;
                
                // Extract Mock Parameters
                FString Persona = ScenarioObj->GetStringField(TEXT("persona"));
                FString Behavior = ScenarioObj->GetStringField(TEXT("current_behavior"));
                FString UserUtterance = ScenarioObj->GetStringField(TEXT("player_input"));
                FString ExpectedOutput = ScenarioObj->GetStringField(TEXT("expected_output")); 

                // System construction combining Persona & Behavior context
                FString SystemPrompt = Persona + TEXT("\n") + Behavior;

                // Sync generation over the DLL / C++ bridge *while on this background thread*
                FString LLMResponse = WeakSubsystem->GenerateFromPrompt(
                    SystemPrompt + TEXT("\nPlayer: ") + UserUtterance + TEXT("\nResponse: ")
                ); 

                // 4. Assemble the exact JSONL log format correctly replacing " quotes avoiding json breaks.
                FString PromptJSON = (SystemPrompt + TEXT(" ") + UserUtterance).Replace(TEXT("\""), TEXT("\\\"")).Replace(TEXT("\n"), TEXT(" "));
                FString ResponseJSON = LLMResponse.Replace(TEXT("\""), TEXT("\\\"")).Replace(TEXT("\n"), TEXT(" "));
                FString ExpectedJSON = ExpectedOutput.Replace(TEXT("\""), TEXT("\\\"")).Replace(TEXT("\n"), TEXT(" "));
                
                FString LogEntry = FString::Printf(TEXT("{\"prompt\": \"%s\", \"response\": \"%s\", \"reference\": \"%s\"}\n"), 
                    *PromptJSON, 
                    *ResponseJSON,
                    *ExpectedJSON);
                
                AllLogs += LogEntry;
            }

            // 5. Finalize the Benchmark IO write down
            FFileHelper::SaveStringToFile(AllLogs, *OutputLogPath, FFileHelper::EEncodingOptions::ForceUTF8);
            
            // Return to Game Thread simply to print the visual alert
            AsyncTask(ENamedThreads::GameThread, [OutputLogPath]() {
                UE_LOG(LogTemp, Warning, TEXT("=== Benchmark Hoàn Tất! Saved: %s ==="), *OutputLogPath);
            });
        }
    });
}
