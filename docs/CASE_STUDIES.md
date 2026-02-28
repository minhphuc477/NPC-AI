# Narrative Case Studies: Contextual Wins and Fails

This document provides a qualitative analysis of the BD-NSCA model's performance, highlighting 5 contextual wins (where the model successfully integrated dynamic game state) and 5 contextual fails (where it struggled).

## Contextual Wins

### Win 1: Immediate Threat Recognition
**Context:** Player approaches the NPC (Guard) while a hostile entity (Goblin) is in the NPC's perception radius.
**Player:** "What's going on here?"
**NPC:** *"Keep your voice down! There's a goblin prowling near the treeline. Draw your weapon if you want to live."*
**Analysis:** The model perfectly integrated the `threat_detected` event from the game state into its response, prioritizing the immediate danger over a generic greeting.

### Win 2: Location-Aware Dialogue
**Context:** NPC (Merchant) is located in the "Frozen Wastes" zone.
**Player:** "Nice weather we're having."
**NPC:** *"Nice? It's freezing! If you think this is nice, you clearly haven't been standing out here all day. Buy a coat or move along."*
**Analysis:** The model correctly utilized the location telemetry to ground the dialogue, recognizing the sarcasm in the player's input based on the environmental context.

### Win 3: Behavior Tree State Reflection
**Context:** NPC (Blacksmith) is currently executing the `BTTask_HammerAnvil` state.
**Player:** "Can you talk for a minute?"
**NPC:** *"Hold on, let me finish this strike! (CLANG!) ... Alright, what do you need? Make it quick, the iron is losing heat."*
**Analysis:** The model dynamically acknowledged its current action (hammering) rather than immediately dropping what it was doing to answer the player.

### Win 4: Memory Integration
**Context:** Player previously helped the NPC (Villager) find their lost ring.
**Player:** "Hello again."
**NPC:** *"Ah, the traveler who found my ring! I'll never forget your kindness. How can I repay you today?"*
**Analysis:** The model successfully retrieved a past interaction from its memory store and seamlessly wove it into the current greeting.

### Win 5: Emotion and Tone Adjustment
**Context:** NPC (Healer) has a current emotion of `fear` (valence -0.8) due to a recent attack.
**Player:** "I need some potions."
**NPC:** *"(Trembling slightly) Yes, yes, of course. Please, just take what you need and hurry. We aren't safe here."*
**Analysis:** The model adapted its tone and added descriptive actions reflecting its current emotional state as provided by the context.

---

## Contextual Fails

### Fail 1: Over-fixation on Single Context Variable
**Context:** NPC (Farmer) has a low health state, but the player is asking about directions.
**Player:** "Which way to the castle?"
**NPC:** *"I'm bleeding! I need a healer! The pain is unbearable! Oh, and the castle is to the north."*
**Analysis:** The formulation forces the model to heavily weight its internal health state, making its response incongruous with the flow of a normal conversation. The transition from panic to giving directions is jarring.

### Fail 2: Hallucinated Game State
**Context:** NPC (Guard) is guarding a gate. No threats are around.
**Player:** "Can I pass?"
**NPC:** *"Not while that dragon is attacking the city! Take cover!"*
**Analysis:** The model hallucinated a dramatic event not present in the dynamic context, likely due to a spurious correlation in the fine-tuning data or the base model's bias towards fantasy tropes.

### Fail 3: Persona Bleed
**Context:** NPC is a lowly, uneducated beggar.
**Player:** "What is the meaning of this curse?"
**NPC:** *"The curse is a manifestation of necrotic energies stemming from the anomalous temporal rift in the eastern quadrant. It systematically degrades cellular structures."*
**Analysis:** The model failed to maintain the requested persona, defaulting to the highly analytical tone of the base model instead of speaking like an uneducated commoner.

### Fail 4: Context Contradiction
**Context:** NPC is located indoors (`location: Tavern`).
**Player:** "Look at the stars tonight."
**NPC:** *"Yes, aren't they beautiful? I love sitting out here and gazing up at them."*
**Analysis:** The model failed to ground its response in the provided location context, agreeing with the player's premise even though they are inside.

### Fail 5: Repetitive Action Descriptions
**Context:** NPC (Merchant) is in the `BTTask_Idle` state.
**Player:** "Show me your wares."
**NPC:** *"(Stands idly) Yes, take a look. (Stands idly) I have many things. (Stands idly)"*
**Analysis:** The model over-indexed on the dynamic context `state: Idle` and unnaturally injected the behavior description multiple times into the response.
