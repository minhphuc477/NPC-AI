// NPCContextExtractor.cpp
// Implementation of dynamic context extraction from UE5

#include "NPCContextExtractor.h"
#include "GameFramework/Character.h"
#include "GameFramework/PlayerController.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "NavigationSystem.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "Perception/AIPerceptionComponent.h"
#include "Perception/AISenseConfig_Sight.h"
#include "Perception/AISenseConfig_Hearing.h"

FNPCDynamicContext UNPCContextExtractor::ExtractContext(
    AActor* NPCActor,
    AAIController* AIController,
    float ScanRadius)
{
    FNPCDynamicContext Context;

    if (!NPCActor)
    {
        UE_LOG(LogTemp, Warning, TEXT("NPCContextExtractor: NPCActor is null"));
        return Context;
    }

    UWorld* World = NPCActor->GetWorld();
    if (!World)
    {
        return Context;
    }

    // Extract NPC identity
    Context.NPCID = NPCActor->GetName();
    
    // Try to get role from actor tags
    if (NPCActor->Tags.Num() > 0)
    {
        Context.NPCRole = NPCActor->Tags[0].ToString();
    }
    else
    {
        Context.NPCRole = TEXT("NPC");
    }

    // Extract location information
    ExtractLocationInfo(NPCActor, Context);

    // Extract Behavior Tree state if AI controller provided
    if (AIController)
    {
        ExtractBehaviorTreeState(AIController, Context);
        ExtractPerceptionInfo(AIController, Context);
    }

    // Extract nearby entities
    ExtractNearbyEntities(NPCActor, ScanRadius, Context);

    // Extract environment info
    ExtractEnvironmentInfo(World, Context);

    return Context;
}

void UNPCContextExtractor::ExtractLocationInfo(
    AActor* NPCActor,
    FNPCDynamicContext& OutContext)
{
    if (!NPCActor)
    {
        return;
    }

    // Get position
    OutContext.Position = NPCActor->GetActorLocation();

    // Detect zone name
    UWorld* World = NPCActor->GetWorld();
    if (World)
    {
        OutContext.ZoneName = DetectZoneName(World, OutContext.Position);
    }

    // Generate location description
    OutContext.LocationName = FString::Printf(
        TEXT("Position (%.0f, %.0f, %.0f) in %s"),
        OutContext.Position.X,
        OutContext.Position.Y,
        OutContext.Position.Z,
        *OutContext.ZoneName
    );
}

void UNPCContextExtractor::ExtractBehaviorTreeState(
    AAIController* AIController,
    FNPCDynamicContext& OutContext)
{
    if (!AIController)
    {
        return;
    }

    // Get Behavior Tree component
    UBehaviorTreeComponent* BTComp = Cast<UBehaviorTreeComponent>(
        AIController->GetBrainComponent()
    );

    if (!BTComp)
    {
        OutContext.CurrentBehavior = TEXT("No Behavior Tree");
        return;
    }

    // Get active node name
    if (BTComp->GetActiveNode())
    {
        OutContext.CurrentBehavior = BTComp->GetActiveNode()->GetNodeName();
    }
    else
    {
        OutContext.CurrentBehavior = TEXT("Idle");
    }

    // Get blackboard component
    UBlackboardComponent* Blackboard = AIController->GetBlackboardComponent();
    if (Blackboard)
    {
        // Extract important blackboard values
        TArray<FName> KeyNames;
        Blackboard->GetAllKeys(KeyNames);

        for (const FName& KeyName : KeyNames)
        {
            FString Value = GetBlackboardValueAsString(Blackboard, KeyName);
            if (!Value.IsEmpty())
            {
                OutContext.BlackboardValues.Add(KeyName.ToString(), Value);
            }
        }

        // Get specific common keys
        if (Blackboard->GetValueAsBool(TEXT("IsAlerted")))
        {
            OutContext.BehaviorState = TEXT("Alerted");
        }
        else if (Blackboard->GetValueAsBool(TEXT("IsPatrolling")))
        {
            OutContext.BehaviorState = TEXT("Patrolling");
        }
        else if (Blackboard->GetValueAsBool(TEXT("IsInCombat")))
        {
            OutContext.BehaviorState = TEXT("In Combat");
        }
        else
        {
            OutContext.BehaviorState = TEXT("Normal");
        }
    }
}

void UNPCContextExtractor::ExtractNearbyEntities(
    AActor* NPCActor,
    float ScanRadius,
    FNPCDynamicContext& OutContext)
{
    if (!NPCActor)
    {
        return;
    }

    UWorld* World = NPCActor->GetWorld();
    if (!World)
    {
        return;
    }

    FVector NPCLocation = NPCActor->GetActorLocation();
    OutContext.NearestPlayerDistance = -1.0f;

    // Phase 9 Fix: AAA FPS Protection
    // Replace GetAllActorsOfClass with spatial overlap query
    TArray<AActor*> NearbyActors;
    TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes;
    ObjectTypes.Add(UEngineTypes::ConvertToObjectType(ECollisionChannel::ECC_Pawn));
    ObjectTypes.Add(UEngineTypes::ConvertToObjectType(ECollisionChannel::ECC_PhysicsBody));
    ObjectTypes.Add(UEngineTypes::ConvertToObjectType(ECollisionChannel::ECC_WorldDynamic));
    
    TArray<AActor*> ActorsToIgnore;
    ActorsToIgnore.Add(NPCActor);

    UKismetSystemLibrary::SphereOverlapActors(
        World,
        NPCLocation,
        ScanRadius,
        ObjectTypes,
        AActor::StaticClass(),
        ActorsToIgnore,
        NearbyActors
    );

    for (AActor* Actor : NearbyActors)
    {
        if (!Actor || Actor == NPCActor)
        {
            continue;
        }

        float Distance = FVector::Dist(NPCLocation, Actor->GetActorLocation());
        if (Distance > ScanRadius)
        {
            continue;
        }

        // Check if it's a player
        ACharacter* Character = Cast<ACharacter>(Actor);
        if (Character && Character->IsPlayerControlled())
        {
            FString PlayerInfo = FString::Printf(
                TEXT("Player (%.0fm away)"),
                Distance / 100.0f  // Convert to meters
            );
            OutContext.NearbyPlayers.Add(PlayerInfo);

            // Track nearest player
            if (OutContext.NearestPlayerDistance < 0 || 
                Distance < OutContext.NearestPlayerDistance)
            {
                OutContext.NearestPlayerDistance = Distance;
            }
        }
        else
        {
            // Other actors
            FString ActorInfo = FString::Printf(
                TEXT("%s (%.0fm away)"),
                *Actor->GetName(),
                Distance / 100.0f
            );
            OutContext.NearbyActors.Add(ActorInfo);
        }
    }
}

void UNPCContextExtractor::ExtractEnvironmentInfo(
    UWorld* World,
    FNPCDynamicContext& OutContext)
{
    if (!World)
    {
        return;
    }

    // Get time of day (simplified - you may have a custom time system)
    FDateTime Now = FDateTime::Now();
    int32 Hour = Now.GetHour();

    if (Hour >= 6 && Hour < 12)
    {
        OutContext.TimeOfDay = TEXT("Morning");
    }
    else if (Hour >= 12 && Hour < 18)
    {
        OutContext.TimeOfDay = TEXT("Afternoon");
    }
    else if (Hour >= 18 && Hour < 22)
    {
        OutContext.TimeOfDay = TEXT("Evening");
    }
    else
    {
        OutContext.TimeOfDay = TEXT("Night");
    }

    // Weather (placeholder - integrate with your weather system)
    OutContext.Weather = TEXT("Clear");
}

FString UNPCContextExtractor::FormatContextAsScenario(
    const FNPCDynamicContext& Context)
{
    FString Scenario;

    // Location
    Scenario += FString::Printf(
        TEXT("Location: %s. "),
        *Context.ZoneName
    );

    // Behavior state
    if (!Context.BehaviorState.IsEmpty())
    {
        Scenario += FString::Printf(
            TEXT("Currently %s. "),
            *Context.BehaviorState
        );
    }

    // Visible actors (Perception)
    if (Context.bCanSeePlayer)
    {
        Scenario += TEXT("I can see the Player. ");
    }
    
    if (Context.VisibleActors.Num() > 0)
    {
        Scenario += TEXT("Visible: ");
        for (const FString& ActorInfo : Context.VisibleActors)
        {
             Scenario += ActorInfo + TEXT(", ");
        }
        Scenario += TEXT(". ");
    }

    // Nearby players (Sensor Audio/Distance fallback)
    if (Context.NearbyPlayers.Num() > 0)
    {
        if (Context.NearestPlayerDistance > 0)
        {
            float DistanceMeters = Context.NearestPlayerDistance / 100.0f;
            if (DistanceMeters < 2.0f)
            {
                Scenario += TEXT("Player is very close. ");
            }
            else if (DistanceMeters < 5.0f)
            {
                Scenario += TEXT("Player is nearby. ");
            }
            else
            {
                Scenario += FString::Printf(
                    TEXT("Player is %.0f meters away. "),
                    DistanceMeters
                );
            }
        }
    }

    // Time and weather
    Scenario += FString::Printf(
        TEXT("Time: %s. Weather: %s."),
        *Context.TimeOfDay,
        *Context.Weather
    );

    return Scenario;
}

FString UNPCContextExtractor::GetBlackboardValueAsString(
    UBlackboardComponent* Blackboard,
    const FName& KeyName)
{
    if (!Blackboard)
    {
        return FString();
    }

    // Try different value types
    UObject* ObjectValue = Blackboard->GetValueAsObject(KeyName);
    if (ObjectValue)
    {
        return ObjectValue->GetName();
    }

    FVector VectorValue = Blackboard->GetValueAsVector(KeyName);
    if (!VectorValue.IsZero())
    {
        return VectorValue.ToString();
    }

    float FloatValue = Blackboard->GetValueAsFloat(KeyName);
    if (FloatValue != 0.0f)
    {
        return FString::SanitizeFloat(FloatValue);
    }

    int32 IntValue = Blackboard->GetValueAsInt(KeyName);
    if (IntValue != 0)
    {
        return FString::FromInt(IntValue);
    }

    bool BoolValue = Blackboard->GetValueAsBool(KeyName);
    if (BoolValue)
    {
        return TEXT("true");
    }

    return FString();
}

FString UNPCContextExtractor::DetectZoneName(
    UWorld* World,
    const FVector& Position)
{
    if (!World)
    {
        return TEXT("Unknown");
    }

    // Phase 9 Fix: AAA FPS Protection
    // Instead of scanning the whole world for 'Zone' tags, do a local sphere overlap
    TArray<AActor*> ZoneActors;
    TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes;
    ObjectTypes.Add(UEngineTypes::ConvertToObjectType(ECollisionChannel::ECC_WorldDynamic));
    ObjectTypes.Add(UEngineTypes::ConvertToObjectType(ECollisionChannel::ECC_WorldStatic));

    TArray<AActor*> ActorsToIgnore;

    UKismetSystemLibrary::SphereOverlapActors(
        World,
        Position,
        5000.0f, // 50 meters
        ObjectTypes,
        AActor::StaticClass(),
        ActorsToIgnore,
        ZoneActors
    );

    for (AActor* ZoneActor : ZoneActors)
    {
        if (ZoneActor && ZoneActor->ActorHasTag(FName(TEXT("Zone"))))
        {
            return ZoneActor->GetName();
        }
    }

    // Fallback: use position-based naming
    if (Position.Z > 1000.0f)
    {
        return TEXT("Highlands");
    }
    else if (Position.Z < -500.0f)
    {
        return TEXT("Underground");
    }
    else
    {
        return TEXT("Village");
    }
}

void UNPCContextExtractor::ExtractPerceptionInfo(
    AAIController* AIController,
    FNPCDynamicContext& OutContext)
{
    if (!AIController)
    {
        return;
    }

    UAIPerceptionComponent* PerceptionComp = AIController->GetPerceptionComponent();
    if (!PerceptionComp)
    {
        return;
    }

    TArray<AActor*> PerceivedActors;
    PerceptionComp->GetCurrentlyPerceivedActors(nullptr, PerceivedActors);

    OutContext.bCanSeePlayer = false;

    for (AActor* Actor : PerceivedActors)
    {
        if (!Actor) continue;

        // Check if it's a player
        ACharacter* Character = Cast<ACharacter>(Actor);
        if (Character && Character->IsPlayerControlled())
        {
            OutContext.bCanSeePlayer = true;
            OutContext.VisibleActors.Add(FString::Printf(TEXT("Player (%s)"), *Actor->GetName()));
        }
        else
        {
            OutContext.VisibleActors.Add(Actor->GetName());
        }
    }
}