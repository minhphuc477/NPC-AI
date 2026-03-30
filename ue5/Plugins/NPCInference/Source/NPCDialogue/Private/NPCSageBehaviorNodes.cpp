// NPCSageBehaviorNodes.cpp
// Minimal functional implementations for SAGE BT nodes.

#include "NPCSageBehaviorNodes.h"

#include "BehaviorTree/BlackboardComponent.h"
#include "Async/Async.h"
#include "Engine/World.h"
#include "Engine/GameInstance.h"
#include "TimerManager.h"
#include "HAL/CriticalSection.h"
#include "Misc/ScopeLock.h"
#include "Misc/Crc.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"
#include "NPCSageBridge.h"
#include "NPCSageBlackboardKeys.h"
#include "NPCSageWorldFactSubsystem.h"

namespace
{
struct FAsyncPythonTaskState
{
    int64 LatestRequestId = 0;
    bool bPending = false;
};

FCriticalSection GAsyncPythonTaskStateMutex;
TMap<const UBTTaskNode*, FAsyncPythonTaskState> GAsyncPythonTaskStates;
FCriticalSection GCrossSyncMutex;
TMap<const UBehaviorTreeComponent*, FString> GLastBroadcastResponse;

int64 BeginAsyncRequest(UBTTaskNode* TaskNode)
{
    if (!TaskNode)
    {
        return 0;
    }
    FScopeLock Lock(&GAsyncPythonTaskStateMutex);
    FAsyncPythonTaskState& State = GAsyncPythonTaskStates.FindOrAdd(TaskNode);
    State.LatestRequestId += 1;
    State.bPending = true;
    return State.LatestRequestId;
}

void CancelAsyncRequest(UBTTaskNode* TaskNode)
{
    if (!TaskNode)
    {
        return;
    }
    FScopeLock Lock(&GAsyncPythonTaskStateMutex);
    if (FAsyncPythonTaskState* State = GAsyncPythonTaskStates.Find(TaskNode))
    {
        State->LatestRequestId += 1;
        State->bPending = false;
        GAsyncPythonTaskStates.Remove(TaskNode);
    }
}

void CompleteAsyncRequest(UBTTaskNode* TaskNode)
{
    if (!TaskNode)
    {
        return;
    }
    FScopeLock Lock(&GAsyncPythonTaskStateMutex);
    GAsyncPythonTaskStates.Remove(TaskNode);
}

bool IsAsyncRequestCurrent(const UBTTaskNode* TaskNode, int64 RequestId)
{
    if (!TaskNode || RequestId <= 0)
    {
        return false;
    }
    FScopeLock Lock(&GAsyncPythonTaskStateMutex);
    const FAsyncPythonTaskState* State = GAsyncPythonTaskStates.Find(TaskNode);
    return State && State->bPending && State->LatestRequestId == RequestId;
}

void ScheduleAsyncTimeout(
    UBehaviorTreeComponent& OwnerComp,
    UBTTaskNode* TaskNode,
    int64 RequestId,
    float TimeoutSeconds
)
{
    if (!TaskNode || RequestId <= 0 || TimeoutSeconds <= 0.0f)
    {
        return;
    }

    UWorld* World = OwnerComp.GetWorld();
    if (!World)
    {
        return;
    }

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTaskNode> WeakTask(TaskNode);
    FTimerDelegate TimeoutDelegate;
    TimeoutDelegate.BindLambda([WeakOwnerComp, WeakTask, RequestId]()
    {
        if (!WeakOwnerComp.IsValid() || !WeakTask.IsValid())
        {
            return;
        }
        UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
        UBTTaskNode* TaskPtr = WeakTask.Get();
        if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
        {
            return;
        }
        CancelAsyncRequest(TaskPtr);
        TaskPtr->FinishLatentTask(*OwnerCompPtr, EBTNodeResult::Failed);
    });

    FTimerHandle TimeoutHandle;
    World->GetTimerManager().SetTimer(TimeoutHandle, TimeoutDelegate, TimeoutSeconds, false);
}

FString ReadBBString(UBlackboardComponent* BB, const FBlackboardKeySelector& Key)
{
    if (!BB || Key.SelectedKeyName.IsNone())
    {
        return FString();
    }
    return BB->GetValueAsString(Key.SelectedKeyName);
}

bool ReadBBBool(UBlackboardComponent* BB, const FBlackboardKeySelector& Key)
{
    if (!BB || Key.SelectedKeyName.IsNone())
    {
        return false;
    }
    return BB->GetValueAsBool(Key.SelectedKeyName);
}

float ReadBBFloat(UBlackboardComponent* BB, const FBlackboardKeySelector& Key)
{
    if (!BB || Key.SelectedKeyName.IsNone())
    {
        return 0.0f;
    }
    return BB->GetValueAsFloat(Key.SelectedKeyName);
}

int32 ReadBBInt(UBlackboardComponent* BB, const FBlackboardKeySelector& Key)
{
    if (!BB || Key.SelectedKeyName.IsNone())
    {
        return 0;
    }
    return BB->GetValueAsInt(Key.SelectedKeyName);
}

void WriteBBString(UBlackboardComponent* BB, const FBlackboardKeySelector& Key, const FString& Value)
{
    if (BB && !Key.SelectedKeyName.IsNone())
    {
        BB->SetValueAsString(Key.SelectedKeyName, Value);
    }
}

void WriteBBBool(UBlackboardComponent* BB, const FBlackboardKeySelector& Key, bool Value)
{
    if (BB && !Key.SelectedKeyName.IsNone())
    {
        BB->SetValueAsBool(Key.SelectedKeyName, Value);
    }
}

void WriteBBFloat(UBlackboardComponent* BB, const FBlackboardKeySelector& Key, float Value)
{
    if (BB && !Key.SelectedKeyName.IsNone())
    {
        BB->SetValueAsFloat(Key.SelectedKeyName, Value);
    }
}

void WriteBBInt(UBlackboardComponent* BB, const FBlackboardKeySelector& Key, int32 Value)
{
    if (BB && !Key.SelectedKeyName.IsNone())
    {
        BB->SetValueAsInt(Key.SelectedKeyName, Value);
    }
}

FString ToRelationshipBucket(float TrustScore)
{
    if (TrustScore <= -0.6f)
    {
        return TEXT("hostile");
    }
    if (TrustScore <= -0.2f)
    {
        return TEXT("wary");
    }
    if (TrustScore < 0.2f)
    {
        return TEXT("neutral");
    }
    if (TrustScore < 0.6f)
    {
        return TEXT("friendly");
    }
    return TEXT("allied");
}

float TrustDeltaFromEvent(const FString& EventToken)
{
    const FString T = EventToken.TrimStartAndEnd().ToLower();
    if (T == TEXT("quest_complete"))
    {
        return 0.3f;
    }
    if (T == TEXT("gift"))
    {
        return 0.1f;
    }
    if (T == TEXT("caught_lying"))
    {
        return -0.2f;
    }
    if (T == TEXT("threatened"))
    {
        return -0.4f;
    }
    if (T == TEXT("attacked_nearby_npc"))
    {
        return -0.6f;
    }
    return 0.0f;
}

FString MoodFromTrustEvent(const FString& EventToken, const FString& FallbackMood)
{
    const FString T = EventToken.TrimStartAndEnd().ToLower();
    if (T == TEXT("threatened") || T == TEXT("attacked_nearby_npc"))
    {
        return TEXT("agitated");
    }
    if (T == TEXT("caught_lying"))
    {
        return TEXT("tense");
    }
    if (T == TEXT("quest_complete") || T == TEXT("gift"))
    {
        return TEXT("calm");
    }
    return FallbackMood.IsEmpty() ? TEXT("calm") : FallbackMood;
}

FString BuildStateHash(const FString& Snapshot)
{
    const uint32 Crc = FCrc::StrCrc32(*Snapshot);
    return FString::Printf(TEXT("%08x"), Crc);
}

FString BuildJsonStatePayload(
    UBlackboardComponent* BB,
    const FName BehaviorStateKeyName,
    const FName LocationKeyName,
    const FBlackboardKeySelector& ActiveQuestPhaseKey,
    const FName NearbyThreatKeyName,
    const FName IsInCombatKeyName,
    const FBlackboardKeySelector& RelationshipScoreKey,
    const FBlackboardKeySelector& MoodStateKey,
    const FBlackboardKeySelector& SessionTurnCountKey,
    const FString& StateHash
)
{
    if (!BB)
    {
        return FString();
    }
    TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("behavior_state"), BB->GetValueAsString(BehaviorStateKeyName));
    Root->SetStringField(TEXT("location"), BB->GetValueAsString(LocationKeyName));
    Root->SetStringField(TEXT("active_quest_phase"), ReadBBString(BB, ActiveQuestPhaseKey));
    Root->SetBoolField(TEXT("nearby_threat"), BB->GetValueAsBool(NearbyThreatKeyName));
    Root->SetBoolField(TEXT("is_in_combat"), BB->GetValueAsBool(IsInCombatKeyName));
    Root->SetStringField(TEXT("relationship_score"), ReadBBString(BB, RelationshipScoreKey));
    Root->SetStringField(TEXT("mood_state"), ReadBBString(BB, MoodStateKey));
    Root->SetNumberField(TEXT("session_turn_count"), ReadBBInt(BB, SessionTurnCountKey));
    Root->SetStringField(TEXT("state_hash"), StateHash);

    FString Out;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Out);
    FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);
    return Out;
}

TArray<FString> ExtractWorldFactCandidates(const FString& Response)
{
    TArray<FString> Clauses;
    Response.ParseIntoArray(Clauses, TEXT("."), true);
    TArray<FString> Facts;
    for (FString Clause : Clauses)
    {
        Clause.TrimStartAndEndInline();
        if (Clause.Len() < 12)
        {
            continue;
        }
        Facts.Add(Clause);
        if (Facts.Num() >= 3)
        {
            break;
        }
    }
    return Facts;
}
} // namespace

UBTService_ThreatMonitor::UBTService_ThreatMonitor()
{
    NodeName = TEXT("Threat Monitor");
    Interval = 0.1f;
    bNotifyTick = true;
    ThreatEventQueueKey.SelectedKeyName = NPCSageBlackboardKeys::ThreatEventQueue;
    StateTransitionFlagKey.SelectedKeyName = NPCSageBlackboardKeys::StateTransitionFlag;
    PrefixCacheValidKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixCacheValid;
    NearbyThreatKey.SelectedKeyName = NPCSageBlackboardKeys::NearbyThreat;
    IsInCombatKey.SelectedKeyName = NPCSageBlackboardKeys::IsInCombat;
}

void UBTService_ThreatMonitor::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return;
    }
    const bool bThreat = ReadBBBool(BB, NearbyThreatKey) || ReadBBBool(BB, IsInCombatKey);
    if (!bThreat)
    {
        return;
    }

    FString Queue = ReadBBString(BB, ThreatEventQueueKey);
    if (!Queue.IsEmpty())
    {
        Queue += TEXT(" | ");
    }
    Queue += FString::Printf(TEXT("threat@%.3f"), FPlatformTime::Seconds());
    WriteBBString(BB, ThreatEventQueueKey, Queue);
    WriteBBBool(BB, StateTransitionFlagKey, true);
    WriteBBBool(BB, PrefixCacheValidKey, false);
}

UBTService_SessionInit::UBTService_SessionInit()
{
    NodeName = TEXT("Session Init Service");
    Interval = 0.2f;
    bNotifyTick = true;
    SessionInitDoneKey.SelectedKeyName = NPCSageBlackboardKeys::SessionInitDone;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    PlayerIdKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerId;
    BehaviorStateKey.SelectedKeyName = NPCSageBlackboardKeys::BehaviorState;
    PlayerQueryKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerQuery;
    LocationKey.SelectedKeyName = NPCSageBlackboardKeys::Location;
    ActiveQuestPhaseKey.SelectedKeyName = NPCSageBlackboardKeys::ActiveQuestPhase;
    GameStateJsonKey.SelectedKeyName = NPCSageBlackboardKeys::GameStateJson;
    EpisodicMemoryHandleKey.SelectedKeyName = NPCSageBlackboardKeys::EpisodicMemoryHandle;
    EpisodicContextKey.SelectedKeyName = NPCSageBlackboardKeys::EpisodicContext;
    EpisodicMemoryFormattedKey.SelectedKeyName = NPCSageBlackboardKeys::EpisodicMemoryFormatted;
    WorldFactsKey.SelectedKeyName = NPCSageBlackboardKeys::WorldFacts;
    SessionTurnCountKey.SelectedKeyName = NPCSageBlackboardKeys::SessionTurnCount;
    PrefixCacheValidKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixCacheValid;
    TrustScoreKey.SelectedKeyName = NPCSageBlackboardKeys::TrustScore;
    RelationshipScoreKey.SelectedKeyName = NPCSageBlackboardKeys::RelationshipScore;
}

void UBTService_SessionInit::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB || ReadBBBool(BB, SessionInitDoneKey))
    {
        return;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString PlayerId = ReadBBString(BB, PlayerIdKey);
    const FString BehaviorState = ReadBBString(BB, BehaviorStateKey);
    const FString Query = ReadBBString(BB, PlayerQueryKey);
    const FString Location = ReadBBString(BB, LocationKey);
    const FString ActiveQuestPhase = ReadBBString(BB, ActiveQuestPhaseKey);
    const FString GameStateJson = ReadBBString(BB, GameStateJsonKey);

    FString EpisodicJson;
    if (UNPCSageBridge::LoadEpisodicMemory(NPCId, PlayerId, BehaviorState, Query, EpisodicTopK, EpisodicJson))
    {
        WriteBBString(BB, EpisodicMemoryHandleKey, EpisodicJson);
        WriteBBString(BB, EpisodicContextKey, EpisodicJson);
        FString Formatted;
        if (UNPCSageBridge::TryGetJsonStringField(EpisodicJson, TEXT("formatted"), Formatted))
        {
            WriteBBString(BB, EpisodicMemoryFormattedKey, Formatted);
        }
    }

    FString FactsJson;
    if (UNPCSageBridge::LoadWorldFacts(NPCId, Location, ActiveQuestPhase, WorldFactsTopK, FactsJson))
    {
        FString Summary;
        if (UNPCSageBridge::TryGetJsonStringField(FactsJson, TEXT("summary"), Summary))
        {
            WriteBBString(BB, WorldFactsKey, Summary);
        }
    }

    FString TrustJson;
    if (UNPCSageBridge::LoadTrustScore(NPCId, PlayerId, TrustJson))
    {
        FString TrustScoreRaw;
        if (UNPCSageBridge::TryGetJsonStringField(TrustJson, TEXT("trust_score"), TrustScoreRaw))
        {
            WriteBBFloat(BB, TrustScoreKey, FCString::Atof(*TrustScoreRaw));
        }
        FString TrustLevel;
        if (UNPCSageBridge::TryGetJsonStringField(TrustJson, TEXT("trust_level"), TrustLevel))
        {
            WriteBBString(BB, RelationshipScoreKey, TrustLevel);
        }
    }

    FString PrefetchJson;
    const bool bPrefetchOk = UNPCSageBridge::PreFetchContext(
        NPCId,
        BehaviorState,
        Location,
        GameStateJson,
        1,
        PrefetchJson
    );
    WriteBBBool(BB, PrefixCacheValidKey, bPrefetchOk);
    WriteBBInt(BB, SessionTurnCountKey, 0);
    WriteBBBool(BB, SessionInitDoneKey, true);
}

UBTService_QuestStateWatcher::UBTService_QuestStateWatcher()
{
    NodeName = TEXT("Quest State Watcher");
    Interval = 0.5f;
    bNotifyTick = true;
    ActiveQuestPhaseKey.SelectedKeyName = NPCSageBlackboardKeys::ActiveQuestPhase;
    QuestPhaseSourceKey.SelectedKeyName = NPCSageBlackboardKeys::QuestPhaseSource;
    StateTransitionFlagKey.SelectedKeyName = NPCSageBlackboardKeys::StateTransitionFlag;
    PrefixCacheValidKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixCacheValid;
}

void UBTService_QuestStateWatcher::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return;
    }
    const FString PreviousPhase = ReadBBString(BB, ActiveQuestPhaseKey);
    const FString LatestPhase = ReadBBString(BB, QuestPhaseSourceKey);
    if (LatestPhase.IsEmpty() || LatestPhase == PreviousPhase)
    {
        return;
    }
    WriteBBString(BB, ActiveQuestPhaseKey, LatestPhase);
    WriteBBBool(BB, StateTransitionFlagKey, true);
    WriteBBBool(BB, PrefixCacheValidKey, false);
}

UBTService_RelationshipTracker::UBTService_RelationshipTracker()
{
    NodeName = TEXT("Relationship Tracker");
    Interval = 2.0f;
    bNotifyTick = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    PlayerIdKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerId;
    SessionIdKey.SelectedKeyName = NPCSageBlackboardKeys::SessionId;
    TrustScoreKey.SelectedKeyName = NPCSageBlackboardKeys::TrustScore;
    RelationshipScoreKey.SelectedKeyName = NPCSageBlackboardKeys::RelationshipScore;
    TrustEventKey.SelectedKeyName = NPCSageBlackboardKeys::TrustEvent;
    MoodStateKey.SelectedKeyName = NPCSageBlackboardKeys::MoodState;
}

void UBTService_RelationshipTracker::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString PlayerId = ReadBBString(BB, PlayerIdKey);
    const FString SessionId = ReadBBString(BB, SessionIdKey);
    const FString EventToken = ReadBBString(BB, TrustEventKey);
    float Trust = FMath::Clamp(ReadBBFloat(BB, TrustScoreKey), -1.0f, 1.0f);

    const float Delta = TrustDeltaFromEvent(EventToken);
    if (FMath::Abs(Delta) > KINDA_SMALL_NUMBER)
    {
        Trust = FMath::Clamp(Trust + Delta, -1.0f, 1.0f);
        WriteBBString(BB, MoodStateKey, MoodFromTrustEvent(EventToken, ReadBBString(BB, MoodStateKey)));
        WriteBBString(BB, TrustEventKey, FString());
    }
    else
    {
        Trust = FMath::Clamp(Trust * FMath::Clamp(DecayFactorPerTick, 0.90f, 1.0f), -1.0f, 1.0f);
    }

    WriteBBFloat(BB, TrustScoreKey, Trust);
    WriteBBString(BB, RelationshipScoreKey, ToRelationshipBucket(Trust));

    FString PersistJson;
    UNPCSageBridge::StoreTrustScore(NPCId, PlayerId, Trust, SessionId, PersistJson);
}

UBTService_CrossNPCSync::UBTService_CrossNPCSync()
{
    NodeName = TEXT("Cross-NPC Sync");
    Interval = 2.0f;
    bNotifyTick = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    NPCResponseKey.SelectedKeyName = NPCSageBlackboardKeys::NPCResponse;
    WorldFactsKey.SelectedKeyName = NPCSageBlackboardKeys::WorldFacts;
}

void UBTService_CrossNPCSync::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return;
    }
    UWorld* World = OwnerComp.GetWorld();
    UGameInstance* GI = World ? World->GetGameInstance() : nullptr;
    UNPCSageWorldFactSubsystem* Subsystem = GI ? GI->GetSubsystem<UNPCSageWorldFactSubsystem>() : nullptr;
    if (!Subsystem)
    {
        return;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString Response = ReadBBString(BB, NPCResponseKey);
    bool bShouldBroadcast = false;
    {
        FScopeLock Lock(&GCrossSyncMutex);
        const FString* Last = GLastBroadcastResponse.Find(&OwnerComp);
        bShouldBroadcast = !Response.IsEmpty() && (!Last || !Last->Equals(Response, ESearchCase::CaseSensitive));
        if (bShouldBroadcast)
        {
            GLastBroadcastResponse.Add(&OwnerComp, Response);
        }
    }
    if (bShouldBroadcast)
    {
        const TArray<FString> Facts = ExtractWorldFactCandidates(Response);
        for (int32 i = 0; i < Facts.Num(); ++i)
        {
            const FString Key = FString::Printf(TEXT("WorldFacts::%s::fact_%d"), *NPCId, i);
            Subsystem->BroadcastFact(Key, Facts[i], FName(*NPCId));
        }
    }

    const FString Summary = Subsystem->BuildSummaryForNPC(FName(*NPCId), MaxSummaryFacts);
    if (!Summary.IsEmpty())
    {
        WriteBBString(BB, WorldFactsKey, Summary);
    }
}

UBTTask_EpisodicMemoryLoad::UBTTask_EpisodicMemoryLoad()
{
    NodeName = TEXT("Episodic Memory Load");
    bCreateNodeInstance = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    PlayerIdKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerId;
    BehaviorStateKey.SelectedKeyName = NPCSageBlackboardKeys::BehaviorState;
    PlayerQueryKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerQuery;
    EpisodicMemoryHandleKey.SelectedKeyName = NPCSageBlackboardKeys::EpisodicMemoryHandle;
    EpisodicMemoryFormattedKey.SelectedKeyName = NPCSageBlackboardKeys::EpisodicMemoryFormatted;
}

EBTNodeResult::Type UBTTask_EpisodicMemoryLoad::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString PlayerId = ReadBBString(BB, PlayerIdKey);
    const FString BehaviorState = ReadBBString(BB, BehaviorStateKey);
    const FString Query = ReadBBString(BB, PlayerQueryKey);
    const int32 LocalTopK = TopK;
    const int64 RequestId = BeginAsyncRequest(this);

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTask_EpisodicMemoryLoad> WeakThis(this);
    ScheduleAsyncTimeout(OwnerComp, this, RequestId, AsyncTimeoutSeconds);

    Async(EAsyncExecution::ThreadPool, [WeakOwnerComp, WeakThis, NPCId, PlayerId, BehaviorState, Query, LocalTopK, RequestId]()
    {
        FString ResultJson;
        const bool bOk = UNPCSageBridge::LoadEpisodicMemory(
            NPCId,
            PlayerId,
            BehaviorState,
            Query,
            LocalTopK,
            ResultJson
        );

        AsyncTask(ENamedThreads::GameThread, [WeakOwnerComp, WeakThis, bOk, ResultJson, RequestId]()
        {
            if (!WeakOwnerComp.IsValid() || !WeakThis.IsValid())
            {
                return;
            }

            UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
            UBTTask_EpisodicMemoryLoad* TaskPtr = WeakThis.Get();
            if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
            {
                return;
            }
            UBlackboardComponent* LocalBB = OwnerCompPtr->GetBlackboardComponent();
            if (LocalBB)
            {
                WriteBBString(LocalBB, TaskPtr->EpisodicMemoryHandleKey, ResultJson);
                FString Formatted;
                if (UNPCSageBridge::TryGetJsonStringField(ResultJson, TEXT("formatted"), Formatted))
                {
                    WriteBBString(LocalBB, TaskPtr->EpisodicMemoryFormattedKey, Formatted);
                }
                else
                {
                    WriteBBString(LocalBB, TaskPtr->EpisodicMemoryFormattedKey, FString());
                }
            }
            CompleteAsyncRequest(TaskPtr);
            TaskPtr->FinishLatentTask(*OwnerCompPtr, bOk ? EBTNodeResult::Succeeded : EBTNodeResult::Failed);
        });
    });

    return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_EpisodicMemoryLoad::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    CancelAsyncRequest(this);
    return Super::AbortTask(OwnerComp, NodeMemory);
}

UBTTask_WorldFactCheck::UBTTask_WorldFactCheck()
{
    NodeName = TEXT("World Fact Check");
    bCreateNodeInstance = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    LocationKey.SelectedKeyName = NPCSageBlackboardKeys::Location;
    ActiveQuestPhaseKey.SelectedKeyName = NPCSageBlackboardKeys::ActiveQuestPhase;
    WorldFactsKey.SelectedKeyName = NPCSageBlackboardKeys::WorldFacts;
}

EBTNodeResult::Type UBTTask_WorldFactCheck::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString Location = ReadBBString(BB, LocationKey);
    const FString ActiveQuestPhase = ReadBBString(BB, ActiveQuestPhaseKey);
    const int32 LocalMaxFacts = MaxFacts;
    const int64 RequestId = BeginAsyncRequest(this);

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTask_WorldFactCheck> WeakThis(this);
    ScheduleAsyncTimeout(OwnerComp, this, RequestId, AsyncTimeoutSeconds);

    Async(EAsyncExecution::ThreadPool, [WeakOwnerComp, WeakThis, NPCId, Location, ActiveQuestPhase, LocalMaxFacts, RequestId]()
    {
        FString ResultJson;
        const bool bOk = UNPCSageBridge::LoadWorldFacts(
            NPCId,
            Location,
            ActiveQuestPhase,
            LocalMaxFacts,
            ResultJson
        );

        AsyncTask(ENamedThreads::GameThread, [WeakOwnerComp, WeakThis, bOk, ResultJson, RequestId]()
        {
            if (!WeakOwnerComp.IsValid() || !WeakThis.IsValid())
            {
                return;
            }

            UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
            UBTTask_WorldFactCheck* TaskPtr = WeakThis.Get();
            if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
            {
                return;
            }
            UBlackboardComponent* LocalBB = OwnerCompPtr->GetBlackboardComponent();
            if (LocalBB)
            {
                FString Summary;
                if (UNPCSageBridge::TryGetJsonStringField(ResultJson, TEXT("summary"), Summary))
                {
                    WriteBBString(LocalBB, TaskPtr->WorldFactsKey, Summary);
                }
                else
                {
                    WriteBBString(LocalBB, TaskPtr->WorldFactsKey, FString());
                }
            }
            CompleteAsyncRequest(TaskPtr);
            TaskPtr->FinishLatentTask(*OwnerCompPtr, bOk ? EBTNodeResult::Succeeded : EBTNodeResult::Failed);
        });
    });

    return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_WorldFactCheck::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    CancelAsyncRequest(this);
    return Super::AbortTask(OwnerComp, NodeMemory);
}

UBTTask_PrefetchNextContext::UBTTask_PrefetchNextContext()
{
    NodeName = TEXT("Prefetch Next Context");
    bCreateNodeInstance = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    BehaviorStateKey.SelectedKeyName = NPCSageBlackboardKeys::BehaviorState;
    LocationKey.SelectedKeyName = NPCSageBlackboardKeys::Location;
    GameStateJsonKey.SelectedKeyName = NPCSageBlackboardKeys::GameStateJson;
    PrefetchedPassagesKey.SelectedKeyName = NPCSageBlackboardKeys::PrefetchedPassages;
    PrefetchResultKey.SelectedKeyName = NPCSageBlackboardKeys::PrefetchResult;
    PrefixCacheValidKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixCacheValid;
}

EBTNodeResult::Type UBTTask_PrefetchNextContext::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString BehaviorState = ReadBBString(BB, BehaviorStateKey);
    const FString Location = ReadBBString(BB, LocationKey);
    const FString GameStateJson = ReadBBString(BB, GameStateJsonKey);
    const int32 LocalTopN = TopPredictedStates;
    const int64 RequestId = BeginAsyncRequest(this);

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTask_PrefetchNextContext> WeakThis(this);
    ScheduleAsyncTimeout(OwnerComp, this, RequestId, AsyncTimeoutSeconds);

    Async(EAsyncExecution::ThreadPool, [WeakOwnerComp, WeakThis, NPCId, BehaviorState, Location, GameStateJson, LocalTopN, RequestId]()
    {
        FString ResultJson;
        const bool bOk = UNPCSageBridge::PreFetchContext(
            NPCId,
            BehaviorState,
            Location,
            GameStateJson,
            LocalTopN,
            ResultJson
        );

        AsyncTask(ENamedThreads::GameThread, [WeakOwnerComp, WeakThis, bOk, ResultJson, RequestId]()
        {
            if (!WeakOwnerComp.IsValid() || !WeakThis.IsValid())
            {
                return;
            }
            UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
            UBTTask_PrefetchNextContext* TaskPtr = WeakThis.Get();
            if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
            {
                return;
            }
            UBlackboardComponent* LocalBB = OwnerCompPtr->GetBlackboardComponent();
            if (LocalBB)
            {
                WriteBBString(LocalBB, TaskPtr->PrefetchResultKey, ResultJson);
                FString Prefetched;
                if (UNPCSageBridge::TryGetJsonStringField(ResultJson, TEXT("prefetched_summary"), Prefetched))
                {
                    WriteBBString(LocalBB, TaskPtr->PrefetchedPassagesKey, Prefetched);
                }
                else
                {
                    WriteBBString(LocalBB, TaskPtr->PrefetchedPassagesKey, ResultJson);
                }
                WriteBBBool(LocalBB, TaskPtr->PrefixCacheValidKey, bOk);
            }
            CompleteAsyncRequest(TaskPtr);
            TaskPtr->FinishLatentTask(*OwnerCompPtr, bOk ? EBTNodeResult::Succeeded : EBTNodeResult::Failed);
        });
    });

    return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_PrefetchNextContext::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    CancelAsyncRequest(this);
    return Super::AbortTask(OwnerComp, NodeMemory);
}

UBTTask_StateTransitionDetector::UBTTask_StateTransitionDetector()
{
    NodeName = TEXT("State Transition Detector");
    LastStateSnapshotKey.SelectedKeyName = NPCSageBlackboardKeys::LastStateSnapshot;
    CurrentStateSnapshotKey.SelectedKeyName = NPCSageBlackboardKeys::CurrentStateSnapshot;
    StateTransitionFlagKey.SelectedKeyName = NPCSageBlackboardKeys::StateTransitionFlag;
    PrefixCacheValidKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixCacheValid;
    StateHashKey.SelectedKeyName = NPCSageBlackboardKeys::StateHash;
    SessionTurnCountKey.SelectedKeyName = NPCSageBlackboardKeys::SessionTurnCount;
    RelationshipScoreKey.SelectedKeyName = NPCSageBlackboardKeys::RelationshipScore;
    MoodStateKey.SelectedKeyName = NPCSageBlackboardKeys::MoodState;
    ActiveQuestPhaseKey.SelectedKeyName = NPCSageBlackboardKeys::ActiveQuestPhase;
    GameStateJsonKey.SelectedKeyName = NPCSageBlackboardKeys::GameStateJson;
    WatchedKeyNames = {
        NPCSageBlackboardKeys::BehaviorState,
        NPCSageBlackboardKeys::Location,
        NPCSageBlackboardKeys::ActiveQuestPhase,
        NPCSageBlackboardKeys::NearbyThreat,
        NPCSageBlackboardKeys::IsInCombat,
        NPCSageBlackboardKeys::RelationshipScore,
        NPCSageBlackboardKeys::MoodState,
    };
}

EBTNodeResult::Type UBTTask_StateTransitionDetector::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }

    FString Snapshot;
    for (const FName& KeyName : WatchedKeyNames)
    {
        if (KeyName.IsNone())
        {
            continue;
        }
        Snapshot += KeyName.ToString();
        Snapshot += TEXT("=");
        Snapshot += BB->GetValueAsString(KeyName);
        Snapshot += TEXT(";");
    }
    if (Snapshot.IsEmpty())
    {
        Snapshot = ReadBBString(BB, CurrentStateSnapshotKey);
    } else {
        WriteBBString(BB, CurrentStateSnapshotKey, Snapshot);
    }

    const FString LastSnapshot = ReadBBString(BB, LastStateSnapshotKey);
    const bool bChanged = !Snapshot.Equals(LastSnapshot, ESearchCase::CaseSensitive);
    if (bChanged)
    {
        WriteBBString(BB, LastStateSnapshotKey, Snapshot);
    }
    const int32 NextTurnCount = FMath::Max(0, ReadBBInt(BB, SessionTurnCountKey) + 1);
    WriteBBInt(BB, SessionTurnCountKey, NextTurnCount);
    const FString StateHash = BuildStateHash(Snapshot);
    WriteBBString(BB, StateHashKey, StateHash);
    WriteBBString(
        BB,
        GameStateJsonKey,
        BuildJsonStatePayload(
            BB,
            NPCSageBlackboardKeys::BehaviorState,
            NPCSageBlackboardKeys::Location,
            ActiveQuestPhaseKey,
            NPCSageBlackboardKeys::NearbyThreat,
            NPCSageBlackboardKeys::IsInCombat,
            RelationshipScoreKey,
            MoodStateKey,
            SessionTurnCountKey,
            StateHash
        )
    );
    WriteBBBool(BB, StateTransitionFlagKey, bChanged);
    WriteBBBool(BB, PrefixCacheValidKey, !bChanged);
    return EBTNodeResult::Succeeded;
}

UBTTask_PrefixCacheInvalidator::UBTTask_PrefixCacheInvalidator()
{
    NodeName = TEXT("Prefix Cache Invalidator");
    bCreateNodeInstance = true;
    StateTransitionFlagKey.SelectedKeyName = NPCSageBlackboardKeys::StateTransitionFlag;
    PrefixCacheValidKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixCacheValid;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    GameStateJsonKey.SelectedKeyName = NPCSageBlackboardKeys::GameStateJson;
    PrefixInvalidationResultKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixInvalidationResult;
}

EBTNodeResult::Type UBTTask_PrefixCacheInvalidator::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }
    if (!ReadBBBool(BB, StateTransitionFlagKey))
    {
        return EBTNodeResult::Succeeded;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString GameStateJson = ReadBBString(BB, GameStateJsonKey);
    WriteBBBool(BB, StateTransitionFlagKey, false);
    WriteBBBool(BB, PrefixCacheValidKey, false);
    const int64 RequestId = BeginAsyncRequest(this);

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTask_PrefixCacheInvalidator> WeakThis(this);
    ScheduleAsyncTimeout(OwnerComp, this, RequestId, AsyncTimeoutSeconds);

    Async(EAsyncExecution::ThreadPool, [WeakOwnerComp, WeakThis, NPCId, GameStateJson, RequestId]()
    {
        FString ResultJson;
        const bool bOk = UNPCSageBridge::InvalidatePrefixCache(
            NPCId,
            GameStateJson,
            ResultJson
        );

        AsyncTask(ENamedThreads::GameThread, [WeakOwnerComp, WeakThis, bOk, ResultJson, RequestId]()
        {
            if (!WeakOwnerComp.IsValid() || !WeakThis.IsValid())
            {
                return;
            }
            UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
            UBTTask_PrefixCacheInvalidator* TaskPtr = WeakThis.Get();
            if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
            {
                return;
            }
            UBlackboardComponent* LocalBB = OwnerCompPtr->GetBlackboardComponent();
            if (LocalBB)
            {
                WriteBBString(LocalBB, TaskPtr->PrefixInvalidationResultKey, ResultJson);
            }
            CompleteAsyncRequest(TaskPtr);
            TaskPtr->FinishLatentTask(*OwnerCompPtr, bOk ? EBTNodeResult::Succeeded : EBTNodeResult::Failed);
        });
    });

    return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_PrefixCacheInvalidator::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    CancelAsyncRequest(this);
    return Super::AbortTask(OwnerComp, NodeMemory);
}

UBTTask_EpisodicMemoryExtract::UBTTask_EpisodicMemoryExtract()
{
    NodeName = TEXT("Episodic Memory Extract");
    bCreateNodeInstance = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    PersonaKey.SelectedKeyName = NPCSageBlackboardKeys::Persona;
    BehaviorStateKey.SelectedKeyName = NPCSageBlackboardKeys::BehaviorState;
    LocationKey.SelectedKeyName = NPCSageBlackboardKeys::Location;
    PlayerInputKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerInput;
    NPCResponseKey.SelectedKeyName = NPCSageBlackboardKeys::NPCResponse;
    SessionIdKey.SelectedKeyName = NPCSageBlackboardKeys::SessionId;
    ExtractResultKey.SelectedKeyName = NPCSageBlackboardKeys::ExtractResult;
}

EBTNodeResult::Type UBTTask_EpisodicMemoryExtract::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString Persona = ReadBBString(BB, PersonaKey);
    const FString BehaviorState = ReadBBString(BB, BehaviorStateKey);
    const FString Location = ReadBBString(BB, LocationKey);
    const FString PlayerInput = ReadBBString(BB, PlayerInputKey);
    const FString NPCResponse = ReadBBString(BB, NPCResponseKey);
    const FString SessionId = ReadBBString(BB, SessionIdKey);
    const int64 RequestId = BeginAsyncRequest(this);

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTask_EpisodicMemoryExtract> WeakThis(this);
    ScheduleAsyncTimeout(OwnerComp, this, RequestId, AsyncTimeoutSeconds);

    Async(EAsyncExecution::ThreadPool, [WeakOwnerComp, WeakThis, NPCId, Persona, BehaviorState, Location, PlayerInput, NPCResponse, SessionId, RequestId]()
    {
        FString ResultJson;
        const bool bOk = UNPCSageBridge::StoreEpisodicMemory(
            NPCId,
            Persona,
            BehaviorState,
            Location,
            PlayerInput,
            NPCResponse,
            SessionId,
            ResultJson
        );

        AsyncTask(ENamedThreads::GameThread, [WeakOwnerComp, WeakThis, bOk, ResultJson, RequestId]()
        {
            if (!WeakOwnerComp.IsValid() || !WeakThis.IsValid())
            {
                return;
            }
            UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
            UBTTask_EpisodicMemoryExtract* TaskPtr = WeakThis.Get();
            if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
            {
                return;
            }
            UBlackboardComponent* LocalBB = OwnerCompPtr->GetBlackboardComponent();
            if (LocalBB)
            {
                WriteBBString(LocalBB, TaskPtr->ExtractResultKey, ResultJson);
            }
            CompleteAsyncRequest(TaskPtr);
            TaskPtr->FinishLatentTask(*OwnerCompPtr, bOk ? EBTNodeResult::Succeeded : EBTNodeResult::Failed);
        });
    });

    return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_EpisodicMemoryExtract::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    CancelAsyncRequest(this);
    return Super::AbortTask(OwnerComp, NodeMemory);
}

UBTTask_OnlinePreferenceLogger::UBTTask_OnlinePreferenceLogger()
{
    NodeName = TEXT("Online Preference Logger");
    bCreateNodeInstance = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    PlayerIdKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerId;
    SessionIdKey.SelectedKeyName = NPCSageBlackboardKeys::SessionId;
    ImplicitFeedbackScoreKey.SelectedKeyName = NPCSageBlackboardKeys::ImplicitFeedbackScore;
    FeedbackOutcomeKey.SelectedKeyName = NPCSageBlackboardKeys::FeedbackOutcome;
    FeedbackLogResultKey.SelectedKeyName = NPCSageBlackboardKeys::FeedbackLogResult;
}

EBTNodeResult::Type UBTTask_OnlinePreferenceLogger::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString PlayerId = ReadBBString(BB, PlayerIdKey);
    const FString SessionId = ReadBBString(BB, SessionIdKey);
    const float Score = ReadBBFloat(BB, ImplicitFeedbackScoreKey);
    const FString Outcome = ReadBBString(BB, FeedbackOutcomeKey);
    const int64 RequestId = BeginAsyncRequest(this);

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTask_OnlinePreferenceLogger> WeakThis(this);
    ScheduleAsyncTimeout(OwnerComp, this, RequestId, AsyncTimeoutSeconds);

    Async(EAsyncExecution::ThreadPool, [WeakOwnerComp, WeakThis, NPCId, PlayerId, SessionId, Score, Outcome, RequestId]()
    {
        FString ResultJson;
        const bool bOk = UNPCSageBridge::LogImplicitFeedback(
            NPCId,
            PlayerId,
            SessionId,
            Score,
            Outcome,
            ResultJson
        );

        AsyncTask(ENamedThreads::GameThread, [WeakOwnerComp, WeakThis, bOk, ResultJson, RequestId]()
        {
            if (!WeakOwnerComp.IsValid() || !WeakThis.IsValid())
            {
                return;
            }
            UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
            UBTTask_OnlinePreferenceLogger* TaskPtr = WeakThis.Get();
            if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
            {
                return;
            }
            UBlackboardComponent* LocalBB = OwnerCompPtr->GetBlackboardComponent();
            if (LocalBB)
            {
                WriteBBString(LocalBB, TaskPtr->FeedbackLogResultKey, ResultJson);
            }
            CompleteAsyncRequest(TaskPtr);
            TaskPtr->FinishLatentTask(*OwnerCompPtr, bOk ? EBTNodeResult::Succeeded : EBTNodeResult::Failed);
        });
    });

    return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_OnlinePreferenceLogger::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    CancelAsyncRequest(this);
    return Super::AbortTask(OwnerComp, NodeMemory);
}

UBTTask_SessionEndPersist::UBTTask_SessionEndPersist()
{
    NodeName = TEXT("Session End Persist");
    bCreateNodeInstance = true;
    NPCIdKey.SelectedKeyName = NPCSageBlackboardKeys::NPCId;
    PlayerIdKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerId;
    SessionIdKey.SelectedKeyName = NPCSageBlackboardKeys::SessionId;
    TrustScoreKey.SelectedKeyName = NPCSageBlackboardKeys::TrustScore;
    SessionInitDoneKey.SelectedKeyName = NPCSageBlackboardKeys::SessionInitDone;
    PrefixCacheValidKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixCacheValid;
    GameStateJsonKey.SelectedKeyName = NPCSageBlackboardKeys::GameStateJson;
    PrefixInvalidationResultKey.SelectedKeyName = NPCSageBlackboardKeys::PrefixInvalidationResult;
}

EBTNodeResult::Type UBTTask_SessionEndPersist::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return EBTNodeResult::Failed;
    }

    const FString NPCId = ReadBBString(BB, NPCIdKey);
    const FString PlayerId = ReadBBString(BB, PlayerIdKey);
    const FString SessionId = ReadBBString(BB, SessionIdKey);
    const float TrustScore = ReadBBFloat(BB, TrustScoreKey);
    const FString GameStateJson = ReadBBString(BB, GameStateJsonKey);
    const bool bLocalResetSessionFlags = bResetSessionFlags;
    const bool bLocalInvalidatePrefix = bInvalidatePrefixOnClose;
    const int64 RequestId = BeginAsyncRequest(this);

    TWeakObjectPtr<UBehaviorTreeComponent> WeakOwnerComp(&OwnerComp);
    TWeakObjectPtr<UBTTask_SessionEndPersist> WeakThis(this);
    ScheduleAsyncTimeout(OwnerComp, this, RequestId, AsyncTimeoutSeconds);

    Async(EAsyncExecution::ThreadPool, [WeakOwnerComp, WeakThis, NPCId, PlayerId, SessionId, TrustScore, GameStateJson, bLocalResetSessionFlags, bLocalInvalidatePrefix, RequestId]()
    {
        FString PersistJson;
        const bool bPersistOk = UNPCSageBridge::StoreTrustScore(
            NPCId,
            PlayerId,
            TrustScore,
            SessionId,
            PersistJson
        );

        bool bInvalidateOk = true;
        FString InvalidateJson;
        if (bLocalInvalidatePrefix)
        {
            bInvalidateOk = UNPCSageBridge::InvalidatePrefixCache(
                NPCId,
                GameStateJson,
                InvalidateJson
            );
        }

        AsyncTask(ENamedThreads::GameThread, [WeakOwnerComp, WeakThis, bPersistOk, PersistJson, bInvalidateOk, InvalidateJson, bLocalResetSessionFlags, RequestId]()
        {
            if (!WeakOwnerComp.IsValid() || !WeakThis.IsValid())
            {
                return;
            }
            UBehaviorTreeComponent* OwnerCompPtr = WeakOwnerComp.Get();
            UBTTask_SessionEndPersist* TaskPtr = WeakThis.Get();
            if (!IsAsyncRequestCurrent(TaskPtr, RequestId))
            {
                return;
            }
            UBlackboardComponent* LocalBB = OwnerCompPtr->GetBlackboardComponent();
            if (LocalBB)
            {
                WriteBBString(LocalBB, TaskPtr->PrefixInvalidationResultKey, bInvalidateOk ? InvalidateJson : PersistJson);
                if (bLocalResetSessionFlags)
                {
                    WriteBBBool(LocalBB, TaskPtr->SessionInitDoneKey, false);
                    WriteBBBool(LocalBB, TaskPtr->PrefixCacheValidKey, false);
                }
            }
            CompleteAsyncRequest(TaskPtr);
            TaskPtr->FinishLatentTask(
                *OwnerCompPtr,
                (bPersistOk && bInvalidateOk) ? EBTNodeResult::Succeeded : EBTNodeResult::Failed
            );
        });
    });

    return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_SessionEndPersist::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    CancelAsyncRequest(this);
    return Super::AbortTask(OwnerComp, NodeMemory);
}

UBTDecorator_ThreatInterrupt::UBTDecorator_ThreatInterrupt()
{
    NodeName = TEXT("Threat Interrupt");
    ThreatEventQueueKey.SelectedKeyName = NPCSageBlackboardKeys::ThreatEventQueue;
    NearbyThreatKey.SelectedKeyName = NPCSageBlackboardKeys::NearbyThreat;
    IsInCombatKey.SelectedKeyName = NPCSageBlackboardKeys::IsInCombat;
    PlayerDistanceKey.SelectedKeyName = NPCSageBlackboardKeys::PlayerDistance;
    NPCHealthKey.SelectedKeyName = NPCSageBlackboardKeys::NPCHealth;
    InterruptFlagKey.SelectedKeyName = NPCSageBlackboardKeys::InterruptFlag;
}

bool UBTDecorator_ThreatInterrupt::CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const
{
    const bool bParent = Super::CalculateRawConditionValue(OwnerComp, NodeMemory);
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return bParent;
    }
    const bool bThreat = ReadBBBool(BB, NearbyThreatKey) || ReadBBBool(BB, IsInCombatKey) || !ReadBBString(BB, ThreatEventQueueKey).IsEmpty();
    const bool bTooFar = ReadBBFloat(BB, PlayerDistanceKey) > MaxPlayerDistance;
    const bool bLowHealth = ReadBBFloat(BB, NPCHealthKey) > 0.0f && ReadBBFloat(BB, NPCHealthKey) < MinHealthRatio;
    const bool bInterrupt = bThreat || bTooFar || bLowHealth;
    WriteBBBool(BB, InterruptFlagKey, bInterrupt);
    return bParent && !bInterrupt;
}

UBTDecorator_ConsistencyGuard::UBTDecorator_ConsistencyGuard()
{
    NodeName = TEXT("Consistency Guard");
    CandidateResponseKey.SelectedKeyName = NPCSageBlackboardKeys::CandidateResponse;
    WorldFactsKey.SelectedKeyName = NPCSageBlackboardKeys::WorldFacts;
    ConsistencyViolationKey.SelectedKeyName = NPCSageBlackboardKeys::ConsistencyViolation;
}

bool UBTDecorator_ConsistencyGuard::CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const
{
    const bool bParent = Super::CalculateRawConditionValue(OwnerComp, NodeMemory);
    UBlackboardComponent* BB = OwnerComp.GetBlackboardComponent();
    if (!BB)
    {
        return bParent;
    }

    const FString Response = ReadBBString(BB, CandidateResponseKey).ToLower();
    const FString FactsRaw = ReadBBString(BB, WorldFactsKey).ToLower();
    const bool bHasNegation = Response.Contains(TEXT(" not ")) || Response.Contains(TEXT(" never "));
    bool bViolation = false;
    if (bHasNegation && !FactsRaw.IsEmpty())
    {
        TArray<FString> Facts;
        FactsRaw.ParseIntoArray(Facts, TEXT(";"), true);
        for (const FString& Fact : Facts)
        {
            FString Trimmed = Fact;
            Trimmed.TrimStartAndEndInline();
            if (Trimmed.IsEmpty())
            {
                continue;
            }
            TArray<FString> Tokens;
            Trimmed.ParseIntoArray(Tokens, TEXT(" "), true);
            if (Tokens.Num() >= 2)
            {
                if (Response.Contains(Tokens[0]) && Response.Contains(Tokens[1]))
                {
                    bViolation = true;
                    break;
                }
            }
        }
    }
    WriteBBBool(BB, ConsistencyViolationKey, bViolation);
    return bParent && !bViolation;
}
