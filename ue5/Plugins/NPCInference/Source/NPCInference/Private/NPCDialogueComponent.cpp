#include "NPCDialogueComponent.h"
#include "NPCInferenceSubsystem.h"
#include "Kismet/GameplayStatics.h"
#include "Async/Async.h"
#include "AIController.h"
#include "GameFramework/Character.h"

// Include the context extractor
// Note: Adjust path if NPCContextExtractor is in a different module
// For now, assuming it's in the same plugin or accessible
#include "NPCContextExtractor.h"

// Sets default values for this component's properties
UNPCDialogueComponent::UNPCDialogueComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
	
	// Auto-generate NPCID from actor name (will be set in BeginPlay)
	NPCID = TEXT("");
	Persona = TEXT("You are a loyal guard.");
	Scenario = TEXT("");  // Will be dynamically extracted
	bUseDynamicContext = true;
	ContextScanRadius = 1000.0f;  // 10 meters
}


// Called when the game starts
void UNPCDialogueComponent::BeginPlay()
{
	Super::BeginPlay();
	
	// Auto-generate NPCID if not set
	if (NPCID.IsEmpty())
	{
		AActor* Owner = GetOwner();
		if (Owner)
		{
			NPCID = Owner->GetName();
		}
	}
	
	// Try to find AI Controller if not set
	if (!AIController)
	{
		AActor* Owner = GetOwner();
		if (Owner)
		{
			// Try to get AI controller from character
			ACharacter* Character = Cast<ACharacter>(Owner);
			if (Character)
			{
				AIController = Cast<AAIController>(Character->GetController());
			}
			
			// Or try to get from pawn
			APawn* Pawn = Cast<APawn>(Owner);
			if (Pawn && !AIController)
			{
				AIController = Cast<AAIController>(Pawn->GetController());
			}
		}
	}
}

FString UNPCDialogueComponent::ExtractDynamicContext()
{
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		UE_LOG(LogTemp, Warning, TEXT("NPCDialogueComponent: No owner actor"));
		return TEXT("Unknown location");
	}
	
	// Extract dynamic context using NPCContextExtractor
	FNPCDynamicContext Context = UNPCContextExtractor::ExtractContext(
		Owner,
		AIController,
		ContextScanRadius
	);
	
	// Format as scenario string
	FString DynamicScenario = UNPCContextExtractor::FormatContextAsScenario(Context);
	
	// Log for debugging
	UE_LOG(LogTemp, Log, TEXT("Dynamic Context: %s"), *DynamicScenario);
	
	return DynamicScenario;
}

FString UNPCDialogueComponent::GetDynamicScenario()
{
	if (bUseDynamicContext)
	{
		return ExtractDynamicContext();
	}
	else
	{
		return Scenario;
	}
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
		UE_LOG(LogTemp, Warning, TEXT("NPCDialogueComponent: Engine not ready."));
		OnResponseGenerated.Broadcast(TEXT("Error: AI Engine not ready."));
		return;
	}

	// Get dynamic context (or static if disabled)
	FString ContextToUse = bUseDynamicContext ? ExtractDynamicContext() : Scenario;
	
	// Capture values by value for async task
	FString SafePersona = Persona;
	FString SafeName = NPCID;
	FString SafeContext = ContextToUse;
	FString SafeInput = PlayerInput;
	
	UE_LOG(LogTemp, Log, TEXT("NPCDialogueComponent: Generating response with context: %s"), *SafeContext);
	
	// Run generation on background thread
	Async(EAsyncExecution::Thread, [this, Subsystem, SafePersona, SafeName, SafeContext, SafeInput]()
	{
		if (!Subsystem) return;

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

