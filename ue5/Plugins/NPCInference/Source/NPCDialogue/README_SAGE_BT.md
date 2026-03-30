# SAGE Behavior Tree Nodes

New UE-side integration files:

- `NPCDialogue.Build.cs`
- `Private/NPCDialogueModule.cpp`
- `Public/NPCSageBridge.h`
- `Private/NPCSageBridge.cpp`
- `Public/NPCSageBlackboardKeys.h`
- `Private/NPCSageBlackboardKeys.cpp`
- `Public/NPCSageBehaviorNodes.h`
- `Private/NPCSageBehaviorNodes.cpp`

Python handler script:

- `scripts/sage_bt_handlers.py`
- `scripts/sage_bt_daemon.py` (persistent request/response worker)

Smoke test checklist:

- `SMOKE_TEST_PLAN.md`

Tree synchronization:

- Canonical module path is `ue5/Plugins/NPCInference/Source/NPCDialogue`.
- Mirror path `ue5/Source/NPCDialogue` is kept in sync via:
  `python scripts/sync_npcdialogue_tree.py --fix`

## Implemented Node Classes

Services:

- `UBTService_ThreatMonitor`
- `UBTService_SessionInit`
- `UBTService_QuestStateWatcher`
- `UBTService_RelationshipTracker`
- `UBTService_CrossNPCSync`

Task execution model:

- Python-backed tasks (`EpisodicMemoryLoad`, `WorldFactCheck`, `PrefixCacheInvalidator`,
  `EpisodicMemoryExtract`, `OnlinePreferenceLogger`) run bridge calls on a thread-pool worker
  and complete via `FinishLatentTask(...)` on the game thread.
- Each async task has an `AsyncTimeoutSeconds` property and abort-safe request IDs to
  prevent stale callbacks from writing blackboard state after timeout/abort.
- `UNPCSageBridge` first attempts the persistent daemon bridge (`storage/runtime/sage_bridge/requests` +
  `storage/runtime/sage_bridge/responses`) and falls back to one-shot `ExecProcess` if daemon startup fails.

Transport hardening:

- All bridge text payloads are base64-encoded on the C++ side (`--*-b64`) and decoded in
  `sage_bt_handlers.py` with plain-arg fallback for compatibility.

Tasks:

- `UBTTask_EpisodicMemoryLoad`
- `UBTTask_WorldFactCheck`
- `UBTTask_PrefetchNextContext`
- `UBTTask_StateTransitionDetector`
- `UBTTask_PrefixCacheInvalidator`
- `UBTTask_EpisodicMemoryExtract`
- `UBTTask_OnlinePreferenceLogger`
- `UBTTask_SessionEndPersist`

Decorators:

- `UBTDecorator_ThreatInterrupt`
- `UBTDecorator_ConsistencyGuard`

## Concrete Blackboard Key Map

Declared in `NPCSageBlackboardKeys.*` and exposed via
`UNPCSageBlackboardKeyLibrary::GetDefaultSageBlackboardSchema()`.

Core identity/session:

- `NPCId` (`String`)
- `PlayerId` (`String`)
- `SessionId` (`String`)

State/context:

- `BehaviorState` (`String`)
- `Location` (`String`)
- `Persona` (`String`)
- `PlayerQuery` (`String`)
- `PlayerInput` (`String`)
- `CandidateResponse` (`String`)
- `NPCResponse` (`String`)

Threat + transitions:

- `ThreatEventQueue` (`String`)
- `NearbyThreat` (`Bool`)
- `IsInCombat` (`Bool`)
- `InterruptFlag` (`Bool`)
- `PlayerDistance` (`Float`)
- `NPCHealth` (`Float`)
- `SessionInitDone` (`Bool`)
- `StateTransitionFlag` (`Bool`)
- `PrefixCacheValid` (`Bool`)
- `GameStateJson` (`String`)
- `PrefixInvalidationResult` (`String`)
- `CurrentStateSnapshot` (`String`)
- `LastStateSnapshot` (`String`)
- `StateHash` (`String`)
- `SessionTurnCount` (`Int`)
- `MoodState` (`String`)
- `TrustScore` (`Float`)
- `RelationshipScore` (`String`)
- `TrustEvent` (`String`)

Quest + world facts:

- `ActiveQuestPhase` (`String`)
- `QuestPhaseSource` (`String`)
- `WorldFacts` (`String`)
- `ConsistencyViolation` (`Bool`)

Episodic memory + feedback:

- `EpisodicMemoryHandle` (`String`)
- `EpisodicContext` (`String`)
- `EpisodicMemoryFormatted` (`String`)
- `ExtractResult` (`String`)
- `PrefetchedPassages` (`String`)
- `PrefetchResult` (`String`)
- `ImplicitFeedbackScore` (`Float`)
- `FeedbackOutcome` (`String`)
- `FeedbackLogResult` (`String`)

## Formatted Episodic Text Wiring

`UBTTask_EpisodicMemoryLoad` now:

1. stores raw handler JSON in `EpisodicMemoryHandle`
2. parses the JSON field `formatted` via `UNPCSageBridge::TryGetJsonStringField`
3. writes prompt-ready text to `EpisodicMemoryFormatted`

## Python Commands

The bridge script supports:

- `invalidate-prefix-cache`
- `warm-prefix-cache`
- `load-episodic`
- `extract-episodic`
- `extract-episodic-interrupt`
- `load-world-facts`
- `prefetch-context`
- `load-trust-score`
- `store-trust-score`
- `log-feedback`

All command outputs are single-line JSON to stdout for easy parsing in UE tasks.
