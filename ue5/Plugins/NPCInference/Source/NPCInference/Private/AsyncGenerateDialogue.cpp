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

	UWorld* World = WorldContext->GetWorld();
	if (!World)
	{
		OnFailed.Broadcast(TEXT("Invalid World"));
		return;
	}

	// 1. Fetch UObjects safely on the Game Thread
	UGameInstance* GameInstance = World->GetGameInstance();
	if (!GameInstance)
	{
		OnFailed.Broadcast(TEXT("No Game Instance"));
		return;
	}

	UNPCInferenceSubsystem* Subsystem = GameInstance->GetSubsystem<UNPCInferenceSubsystem>();
	if (!Subsystem || !Subsystem->IsEngineReady())
	{
		OnFailed.Broadcast(TEXT("NPC Engine not ready"));
		return;
	}

	// 2. Capture weak pointer to pass to background thread
	TWeakObjectPtr<UNPCInferenceSubsystem> WeakSubsystem(Subsystem);
	TWeakObjectPtr<UAsyncGenerateDialogue> WeakThis(this);

	// 3. Execute on background thread
	Async(EAsyncExecution::Thread, [WeakThis, WeakSubsystem]()
	{
		if (WeakThis.IsValid())
		{
			WeakThis->ExecuteGeneration(WeakSubsystem);
		}
	});
}

void UAsyncGenerateDialogue::ExecuteGeneration(TWeakObjectPtr<UNPCInferenceSubsystem> WeakSubsystem)
{
	// 4. Validate weak pointer safely *inside* the background thread
	if (!WeakSubsystem.IsValid())
	{
		TWeakObjectPtr<UAsyncGenerateDialogue> WeakThis(this);
		AsyncTask(ENamedThreads::GameThread, [WeakThis]()
		{
			if (WeakThis.IsValid())
			{
				WeakThis->OnFailed.Broadcast(TEXT("Subsystem destroyed during generation"));
			}
		});
		return;
	}

	// Generate dialogue (this is the slow part, runs on background thread)
	FString Response = WeakSubsystem->GenerateDialogue(Prompt_System, Prompt_Name, Prompt_Context, Prompt_Input);

	// Phase 9 Fix: AAA Dangling Pointer Protection
	// Return to game thread for callback safely using WeakPtr
	TWeakObjectPtr<UAsyncGenerateDialogue> WeakThis(this);
	AsyncTask(ENamedThreads::GameThread, [WeakThis, Response]()
	{
		if (WeakThis.IsValid())
		{
			WeakThis->OnGenerationComplete(Response);
		}
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
