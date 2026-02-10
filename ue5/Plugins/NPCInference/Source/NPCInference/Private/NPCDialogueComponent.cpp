#include "NPCDialogueComponent.h"
#include "NPCInferenceSubsystem.h"
#include "Kismet/GameplayStatics.h"
#include "Async/Async.h"

// Sets default values for this component's properties
UNPCDialogueComponent::UNPCDialogueComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
	NPCID = TEXT("Guard_1");
	Persona = TEXT("You are a loyal guard.");
	Scenario = TEXT("At the village gate.");
}


// Called when the game starts
void UNPCDialogueComponent::BeginPlay()
{
	Super::BeginPlay();
}

void UNPCDialogueComponent::RequestResponse(const FString& PlayerInput)
{
	if (PlayerInput.IsEmpty())
	{
		UE_LOG(LogTemp, Warning, TEXT("NPCDialogueComponent: PlayerInput is empty."));
		return;
	}

	UGameInstance* GI = UGameplayStatics::GetGameInstance(this);
	if (!GI)
	{
		UE_LOG(LogTemp, Error, TEXT("NPCDialogueComponent: GameInstance not found."));
		return;
	}

	UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>();
	if (!Subsystem)
	{
		UE_LOG(LogTemp, Error, TEXT("NPCDialogueComponent: NPCInferenceSubsystem not found."));
		return;
	}

	if (!Subsystem->IsEngineReady())
	{
		// Try to initialize if not ready? Or just fail.
		// For now fail gracefully.
		UE_LOG(LogTemp, Warning, TEXT("NPCDialogueComponent: Engine not ready."));
		OnResponseGenerated.Broadcast(TEXT("Error: AI Engine not ready."));
		return;
	}

	// Capture values by value for async task
	FString SafePersona = Persona;
	FString SafeName = NPCID;
	FString SafeContext = Scenario;
	FString SafeInput = PlayerInput;
	
	// Run generation on background thread
	// Use Async from Async/Async.h
	Async(EAsyncExecution::Thread, [this, Subsystem, SafePersona, SafeName, SafeContext, SafeInput]()
	{
		if (!Subsystem) return; // Paranoia check

		// This runs on background thread
		FString Response = Subsystem->GenerateDialogue(SafePersona, SafeName, SafeContext, SafeInput);

		// Broadcast on Game Thread
		AsyncTask(ENamedThreads::GameThread, [this, Response]()
		{
			// Verify we are still valid (not destroyed)
			if (IsValid(this))
			{
				OnResponseGenerated.Broadcast(Response);
			}
		});
	});
}
