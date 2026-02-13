// AsyncGenerateDialogue.cpp - Implementation of Async Dialogue Generation

#include "AsyncGenerateDialogue.h"
#include "NPCInferenceSubsystem.h"
#include "Engine/GameInstance.h"
#include "Async/Async.h"

UAsyncGenerateDialogue* UAsyncGenerateDialogue::AsyncGenerateDialogue(
	UObject* WorldContextObject,
	const FString& SystemPrompt,
	const FString& NPCName,
	const FString& Context,
	const FString& PlayerInput)
{
	UAsyncGenerateDialogue* Action = NewObject<UAsyncGenerateDialogue>();
	Action->WorldContext = WorldContextObject;
	Action->Prompt_System = SystemPrompt;
	Action->Prompt_Name = NPCName;
	Action->Prompt_Context = Context;
	Action->Prompt_Input = PlayerInput;
	return Action;
}

void UAsyncGenerateDialogue::Activate()
{
	if (!WorldContext)
	{
		OnFailed.Broadcast(TEXT("Invalid World Context"));
		return;
	}

	// Execute on background thread
	Async(EAsyncExecution::Thread, [this]()
	{
		ExecuteGeneration();
	});
}

void UAsyncGenerateDialogue::ExecuteGeneration()
{
	// Get subsystem on game thread
	UGameInstance* GameInstance = WorldContext->GetWorld()->GetGameInstance();
	if (!GameInstance)
	{
		AsyncTask(ENamedThreads::GameThread, [this]()
		{
			OnFailed.Broadcast(TEXT("No Game Instance"));
		});
		return;
	}

	UNPCInferenceSubsystem* Subsystem = GameInstance->GetSubsystem<UNPCInferenceSubsystem>();
	if (!Subsystem || !Subsystem->IsEngineReady())
	{
		AsyncTask(ENamedThreads::GameThread, [this]()
		{
			OnFailed.Broadcast(TEXT("NPC Engine not ready"));
		});
		return;
	}

	// Generate dialogue (this is the slow part, runs on background thread)
	FString Response = Subsystem->GenerateDialogue(Prompt_System, Prompt_Name, Prompt_Context, Prompt_Input);

	// Return to game thread for callback
	AsyncTask(ENamedThreads::GameThread, [this, Response]()
	{
		OnGenerationComplete(Response);
	});
}

void UAsyncGenerateDialogue::OnGenerationComplete(const FString& Response)
{
	if (Response.IsEmpty() || Response.StartsWith(TEXT("Error")))
	{
		OnFailed.Broadcast(Response);
	}
	else
	{
		OnComplete.Broadcast(Response);
	}
}
