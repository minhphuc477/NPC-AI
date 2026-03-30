#include "NPCDialogueComponent.h"
#include "NPCInferenceSubsystem.h"
#include "NPCConversationSubsystem.h"
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
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.bStartWithTickEnabled = false; // Phase 8: Only tick when generating
	
	// Auto-generate NPCID from actor name (will be set in BeginPlay)
	NPCID = TEXT("");
	Persona = TEXT("You are a loyal guard.");
	Scenario = TEXT("");  // Will be dynamically extracted
	bUseDynamicContext = true;
	ContextScanRadius = 1000.0f;  // 10 meters
}

void UNPCDialogueComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// Phase 8: Stale Context Prevention (Late Binding Check)
	if (bIsGeneratingResponse && CurrentInteractingPlayer)
	{
		if (AActor* Owner = GetOwner())
		{
			float Distance = FVector::Dist(Owner->GetActorLocation(), CurrentInteractingPlayer->GetActorLocation());
			if (Distance > ContextScanRadius * 1.5f) // Add a little buffer
			{
				UE_LOG(LogTemp, Warning, TEXT("NPC %s: Player walked away. Canceling generation to prevent stale context."), *NPCID);
				
				if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
				{
					if (UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>())
					{
						Subsystem->CancelGeneration(NPCID);
					}
				}
				
				bIsGeneratingResponse = false;
				SetComponentTickEnabled(false);
			}
		}
	}
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

	// Register with Conversation Subsystem (Phase 13)
	if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
	{
		if (UNPCConversationSubsystem* ConversationSys = GI->GetSubsystem<UNPCConversationSubsystem>())
		{
			ConversationSys->RegisterNPC(this);
		}
	}
}

void UNPCDialogueComponent::BroadcastToNearby(const FString& Message)
{
	if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
	{
		if (UNPCConversationSubsystem* ConversationSys = GI->GetSubsystem<UNPCConversationSubsystem>())
		{
			ConversationSys->BroadcastMessage(this, Message, 1000.0f); // Default 10m radius
		}
	}
}

void UNPCDialogueComponent::ReceiveBroadcast(const FString& SenderName, const FString& Message)
{
    // Fix: Route heard speech into the NPC's memory so it influences future dialogue
    HearGossip(Message, SenderName);
    
    UE_LOG(LogTemp, Log, TEXT("NPC %s heard '%s' say: %s"), *NPCID, *SenderName, *Message);
}

void UNPCDialogueComponent::HearGossip(const FString& GossipText, const FString& SourceName)
{
    if (GossipText.IsEmpty()) return;
    
    if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
    {
        if (UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>())
        {
            Subsystem->ReceiveGossip(GossipText, SourceName);
            UE_LOG(LogTemp, Log, TEXT("NPC %s ingested gossip from %s into memory."), *NPCID, *SourceName);
        }
    }
}

FString UNPCDialogueComponent::ShareGossip()
{
    if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
    {
        if (UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>())
        {
            return Subsystem->ExtractGossip();
        }
    }
    return TEXT("");
}

void UNPCDialogueComponent::Sleep()
{
    if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
    {
        if (UNPCInferenceSubsystem* Subsystem = GI->GetSubsystem<UNPCInferenceSubsystem>())
        {
            UE_LOG(LogTemp, Log, TEXT("NPC %s: Triggering sleep/memory consolidation..."), *NPCID);
            Subsystem->TriggerSleepMode();
        }
    }
}

void UNPCDialogueComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
	{
		if (UNPCConversationSubsystem* ConversationSys = GI->GetSubsystem<UNPCConversationSubsystem>())
		{
			ConversationSys->UnregisterNPC(this);
		}
	}
	Super::EndPlay(EndPlayReason);
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
	
	// Phase 8: Setup for Late Binding Check
	CurrentInteractingPlayer = UGameplayStatics::GetPlayerPawn(this, 0); // Assuming local player 0
	if (CurrentInteractingPlayer)
	{
		bIsGeneratingResponse = true;
		SetComponentTickEnabled(true);
	}

	// Create delegate for completion
	FOnDialogueGenerated OnComplete;
	OnComplete.BindDynamic(this, &UNPCDialogueComponent::OnAsyncResponseReceived);

	Subsystem->GenerateDialogueAsync(Persona, NPCID, ContextToUse, PlayerInput, OnComplete);
}

void UNPCDialogueComponent::OnAsyncResponseReceived(const FString& Response)
{
	bIsGeneratingResponse = false;
	SetComponentTickEnabled(false);
	
	if (!Response.Contains(TEXT("Error:")))
	{
		// Fix: Auto-remember what the NPC said to give it memory of its own dialogue
		FString GossipEntry = FString::Printf(TEXT("[%s said]: %s"), *NPCID, *Response);
		HearGossip(GossipEntry, NPCID);
	}
	
	OnResponseGenerated.Broadcast(Response);
}

