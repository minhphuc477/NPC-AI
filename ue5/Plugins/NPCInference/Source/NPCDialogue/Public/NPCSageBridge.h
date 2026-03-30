// NPCSageBridge.h
// Thin UE5 -> Python bridge for SAGE behavior-tree tasks.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "NPCSageBridge.generated.h"

UCLASS()
class NPCDIALOGUE_API UNPCSageBridge : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static void ShutdownBridgeDaemon();

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool InvalidatePrefixCache(
        const FString& NPCId,
        const FString& GameStateJson,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool LoadEpisodicMemory(
        const FString& NPCId,
        const FString& PlayerId,
        const FString& BehaviorState,
        const FString& Query,
        int32 TopK,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool StoreEpisodicMemory(
        const FString& NPCId,
        const FString& Persona,
        const FString& BehaviorState,
        const FString& Location,
        const FString& PlayerInput,
        const FString& NPCResponse,
        const FString& SessionId,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool LoadWorldFacts(
        const FString& NPCId,
        const FString& Location,
        const FString& ActiveQuestPhase,
        int32 MaxFacts,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool LogImplicitFeedback(
        const FString& NPCId,
        const FString& PlayerId,
        const FString& SessionId,
        float Score,
        const FString& Outcome,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool PreFetchContext(
        const FString& NPCId,
        const FString& CurrentBehaviorState,
        const FString& Location,
        const FString& GameStateJson,
        int32 TopPredictedStates,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool LoadTrustScore(
        const FString& NPCId,
        const FString& PlayerId,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintCallable, Category = "NPC|SAGE")
    static bool StoreTrustScore(
        const FString& NPCId,
        const FString& PlayerId,
        float TrustScore,
        const FString& SessionId,
        FString& OutResultJson,
        const FString& PythonExe = TEXT("python"),
        const FString& HandlerScript = TEXT("scripts/sage_bt_handlers.py")
    );

    UFUNCTION(BlueprintPure, Category = "NPC|SAGE")
    static bool TryGetJsonStringField(
        const FString& JsonPayload,
        const FString& FieldName,
        FString& OutValue
    );

private:
    static bool ExecuteHandler(
        const TArray<FString>& Argv,
        FString& OutStdOut,
        FString& OutStdErr,
        int32& OutReturnCode,
        const FString& PythonExe,
        const FString& HandlerScript
    );

    static bool ExecuteHandlerViaDaemon(
        const TArray<FString>& Argv,
        FString& OutStdOut,
        FString& OutStdErr,
        int32& OutReturnCode,
        const FString& PythonExe,
        const FString& HandlerScript
    );

    static bool EnsureDaemonRunning(
        const FString& PythonExe,
        const FString& HandlerScript
    );

    static void CleanupStaleIpcFiles();

    static FString ResolveHandlerScriptPath(const FString& HandlerScript);

    static FString BuildCommandLineArgs(const TArray<FString>& Argv);

    static FString BuildFailurePayload(
        const FString& ErrorCode,
        const FString& StdErr,
        int32 ReturnCode,
        const FString& StdOut
    );
};
