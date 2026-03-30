// NPCSageBehaviorNodes.h
// SAGE-aligned BT services/tasks/decorators for UE5 integration.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTService.h"
#include "BehaviorTree/BTTaskNode.h"
#include "BehaviorTree/BTDecorator.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_Bool.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_String.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_Float.h"
#include "NPCSageBehaviorNodes.generated.h"

UCLASS()
class NPCDIALOGUE_API UBTService_ThreatMonitor : public UBTService
{
    GENERATED_BODY()

public:
    UBTService_ThreatMonitor();

    virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ThreatEventQueueKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector StateTransitionFlagKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixCacheValidKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NearbyThreatKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector IsInCombatKey;
};

UCLASS()
class NPCDIALOGUE_API UBTService_SessionInit : public UBTService
{
    GENERATED_BODY()

public:
    UBTService_SessionInit();

    virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionInitDoneKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector BehaviorStateKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerQueryKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector LocationKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ActiveQuestPhaseKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector GameStateJsonKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector EpisodicMemoryHandleKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector EpisodicContextKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector EpisodicMemoryFormattedKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector WorldFactsKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionTurnCountKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixCacheValidKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector TrustScoreKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector RelationshipScoreKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    int32 EpisodicTopK = 5;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    int32 WorldFactsTopK = 8;
};

UCLASS()
class NPCDIALOGUE_API UBTService_QuestStateWatcher : public UBTService
{
    GENERATED_BODY()

public:
    UBTService_QuestStateWatcher();

    virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ActiveQuestPhaseKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector QuestPhaseSourceKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector StateTransitionFlagKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixCacheValidKey;
};

UCLASS()
class NPCDIALOGUE_API UBTService_RelationshipTracker : public UBTService
{
    GENERATED_BODY()

public:
    UBTService_RelationshipTracker();

    virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector TrustScoreKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector RelationshipScoreKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector TrustEventKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector MoodStateKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float DecayFactorPerTick = 0.995f;
};

UCLASS()
class NPCDIALOGUE_API UBTService_CrossNPCSync : public UBTService
{
    GENERATED_BODY()

public:
    UBTService_CrossNPCSync();

    virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCResponseKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector WorldFactsKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    int32 MaxSummaryFacts = 8;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_EpisodicMemoryLoad : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_EpisodicMemoryLoad();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector BehaviorStateKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerQueryKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector EpisodicMemoryHandleKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector EpisodicMemoryFormattedKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    int32 TopK = 3;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float AsyncTimeoutSeconds = 2.5f;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_WorldFactCheck : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_WorldFactCheck();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector LocationKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ActiveQuestPhaseKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector WorldFactsKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    int32 MaxFacts = 8;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float AsyncTimeoutSeconds = 2.5f;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_PrefetchNextContext : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_PrefetchNextContext();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector BehaviorStateKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector LocationKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector GameStateJsonKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefetchedPassagesKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefetchResultKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixCacheValidKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    int32 TopPredictedStates = 2;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float AsyncTimeoutSeconds = 1.5f;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_StateTransitionDetector : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_StateTransitionDetector();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector LastStateSnapshotKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector CurrentStateSnapshotKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector StateTransitionFlagKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixCacheValidKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector StateHashKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionTurnCountKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector RelationshipScoreKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector MoodStateKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ActiveQuestPhaseKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector GameStateJsonKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    TArray<FName> WatchedKeyNames;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_PrefixCacheInvalidator : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_PrefixCacheInvalidator();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector StateTransitionFlagKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixCacheValidKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector GameStateJsonKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixInvalidationResultKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float AsyncTimeoutSeconds = 1.0f;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_EpisodicMemoryExtract : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_EpisodicMemoryExtract();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PersonaKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector BehaviorStateKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector LocationKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerInputKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCResponseKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ExtractResultKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float AsyncTimeoutSeconds = 2.5f;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_OnlinePreferenceLogger : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_OnlinePreferenceLogger();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ImplicitFeedbackScoreKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector FeedbackOutcomeKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector FeedbackLogResultKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float AsyncTimeoutSeconds = 1.5f;
};

UCLASS()
class NPCDIALOGUE_API UBTTask_SessionEndPersist : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_SessionEndPersist();

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionIdKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector TrustScoreKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector SessionInitDoneKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixCacheValidKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector GameStateJsonKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PrefixInvalidationResultKey;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    bool bResetSessionFlags = true;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    bool bInvalidatePrefixOnClose = true;

    UPROPERTY(EditAnywhere, Category = "SAGE")
    float AsyncTimeoutSeconds = 1.5f;
};

UCLASS()
class NPCDIALOGUE_API UBTDecorator_ThreatInterrupt : public UBTDecorator
{
    GENERATED_BODY()

public:
    UBTDecorator_ThreatInterrupt();

protected:
    virtual bool CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const override;

public:
    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ThreatEventQueueKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NearbyThreatKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector IsInCombatKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector PlayerDistanceKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector NPCHealthKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector InterruptFlagKey;

    UPROPERTY(EditAnywhere, Category = "Policy")
    float MaxPlayerDistance = 3500.0f;

    UPROPERTY(EditAnywhere, Category = "Policy")
    float MinHealthRatio = 0.20f;
};

UCLASS()
class NPCDIALOGUE_API UBTDecorator_ConsistencyGuard : public UBTDecorator
{
    GENERATED_BODY()

public:
    UBTDecorator_ConsistencyGuard();

protected:
    virtual bool CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const override;

public:
    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector CandidateResponseKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector WorldFactsKey;

    UPROPERTY(EditAnywhere, Category = "Blackboard")
    FBlackboardKeySelector ConsistencyViolationKey;
};
