# NPCDialogue UE Smoke Test Plan

## Scope

Validate BT node behavior for:

- async non-blocking task execution
- timeout handling
- abort/cancel safety
- world-fact contract (`summary` string)
- prefix invalidation path

## Preconditions

1. `scripts/sage_bt_handlers.py` is reachable from UE project root.
2. Python environment has required packages for handler dependencies.
3. Blackboard includes SAGE keys from `UNPCSageBlackboardKeyLibrary`.
4. BT includes:
   - `UBTService_SessionInit`
   - `UBTService_RelationshipTracker`
   - `UBTService_CrossNPCSync`
   - `UBTTask_EpisodicMemoryLoad`
   - `UBTTask_WorldFactCheck`
   - `UBTTask_PrefetchNextContext`
   - `UBTTask_PrefixCacheInvalidator`
   - `UBTTask_EpisodicMemoryExtract`
   - `UBTTask_OnlinePreferenceLogger`
   - `UBTTask_SessionEndPersist`
   - `UBTDecorator_ConsistencyGuard`
   - `UBTDecorator_ThreatInterrupt`
5. Daemon script exists at `scripts/sage_bt_daemon.py`.

## Test Cases

### 1. Non-blocking execution

1. Start PIE with stat unit / stat game enabled.
2. Trigger each Python-backed task once.
3. Confirm no frame hitch spikes during task execution.
4. Confirm task transitions from `InProgress` to `Succeeded/Failed` normally.

Expected:

- Game thread frame time remains stable.
- No long stall synchronized to task invocation.
- First call may include daemon startup overhead; subsequent calls should avoid process-spawn spikes.

### 2. Timeout path

1. Set `AsyncTimeoutSeconds = 0.05` on one task (e.g., `WorldFactCheck`).
2. Artificially delay handler (temporary sleep in Python or heavy input).
3. Trigger task.

Expected:

- Task fails via timeout (no hang).
- Later worker callback does not overwrite blackboard keys.

### 3. Abort safety

1. Trigger a Python-backed task.
2. Immediately force BT branch switch/abort (e.g., set threat flag).
3. Observe blackboard keys after abort.

Expected:

- Task aborts cleanly.
- Stale callback does not call success path or mutate result keys post-abort.

### 4. World-fact summary contract

1. Ensure handler world facts file contains multiple facts.
2. Trigger `WorldFactCheck`.
3. Inspect `WorldFacts` blackboard value.
4. Trigger `ConsistencyGuard` with a response containing negation.

Expected:

- `WorldFacts` stores semicolon-delimited summary text.
- Decorator token checks operate on summary text (not raw JSON blob).

### 5. Prefix invalidation end-to-end

1. Trigger `StateTransitionDetector` so `StateTransitionFlag=true`.
2. Trigger `PrefixCacheInvalidator`.
3. Verify `storage/artifacts/gspe/prefix_cache_invalidate_events.jsonl` gets an event.
4. In GSPE runtime test, verify cache invalidation counter increases after event consumption.

Expected:

- Invalidation event logged with `state_hash`.
- Runtime cache entry for matching state is evicted.

### 6. Session init + trust + prefetch

1. Start a new dialogue session.
2. Verify `SessionInitDone=true` after first tick.
3. Verify hydration keys are written:
   - `EpisodicContext`
   - `WorldFacts`
   - `TrustScore`
   - `RelationshipScore`
4. Trigger `PrefetchNextContext`.
5. Verify:
   - `PrefetchResult` contains `predicted_states`.
   - `PrefetchedPassages` is non-empty.
   - `PrefixCacheValid=true` on success.

### 7. Session-end trust persistence

1. Set `TrustScore` on blackboard to a non-zero value (e.g. `0.45`).
2. Execute `UBTTask_SessionEndPersist` at dialogue close.
3. Verify task succeeds and writes `PrefixInvalidationResult`.
4. Re-open dialogue and confirm `SessionInit` reloads same trust bucket from store.

## Pass Criteria

- No game-thread stalls attributable to handler calls.
- Timeout and abort cases complete deterministically.
- Blackboard data contracts match consumer expectations.
- Prefix invalidation events are produced and consumed.
