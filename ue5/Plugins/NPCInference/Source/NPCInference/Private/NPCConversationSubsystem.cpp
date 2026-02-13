#include "NPCConversationSubsystem.h"
#include "NPCDialogueComponent.h"
#include "Kismet/GameplayStatics.h"

void UNPCConversationSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	ActiveNPCs.Empty();
}

void UNPCConversationSubsystem::Deinitialize()
{
	ActiveNPCs.Empty();
	Super::Deinitialize();
}

void UNPCConversationSubsystem::RegisterNPC(UNPCDialogueComponent* NPC)
{
	if (NPC && !ActiveNPCs.Contains(NPC))
	{
		ActiveNPCs.Add(NPC);
	}
}

void UNPCConversationSubsystem::UnregisterNPC(UNPCDialogueComponent* NPC)
{
	if (NPC)
	{
		ActiveNPCs.Remove(NPC);
	}
}

void UNPCConversationSubsystem::BroadcastMessage(UNPCDialogueComponent* Sender, const FString& Message, float Radius)
{
	if (!Sender) return;

	FVector SenderLocation = Sender->GetOwner()->GetActorLocation();
	FString SenderName = Sender->GetOwner()->GetName(); 
    // Ideally get a "Character Name" property, but Actor Name works for now

	float RadiusSq = Radius * Radius;

	for (UNPCDialogueComponent* NPC : ActiveNPCs)
	{
		// Don't hear yourself
		if (NPC == Sender || !NPC->GetOwner()) continue;

		float DistSq = FVector::DistSquared(SenderLocation, NPC->GetOwner()->GetActorLocation());
		if (DistSq <= RadiusSq)
		{
			// Send the message to this NPC
            // Assumes ReceiveBroadcast exists (we will add it next)
			NPC->ReceiveBroadcast(SenderName, Message);
		}
	}
}
