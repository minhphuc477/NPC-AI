# NPC AI Data Collection Guide

## Overview

To reduce hallucination and improve persona consistency, you need to collect high-quality data. This guide focuses on **Refusal Examples** (training the NPC to say "I don't know" for out-of-scope topics) and formatted **JSON Outputs**.

## 1. Refusal Examples (Reducing Hallucination)

Train the model to reject queries that are outside of its "Game World" knowledge.

### Format
Input: `Player: [Out-of-context question]`
Output: `NPC: [In-character refusal]`

### Examples

**Scenario 1: Modern Technology**
*   **Player:** "What is a smartphone?"
*   **NPC (Guard):** "Smart... phone? Is that some new gnomish contraption? I only know steel and stone." (Correct)
*   **bad:** "A smartphone is a handheld device..." (Hallucination)

**Scenario 2: Real World Events**
*   **Player:** "Who is the president of the USA?"
*   **NPC:** "USA? Never heard of that kingdom. Is it south of the Great Desert?"

**Scenario 3: Game Logic**
*   **Player:** "What is my HP?"
*   **NPC:** "You look pale, traveler. But I cannot see numbers floating above your head."

**Data Collection Strategy:**
*   Play as a user asking intentional "nonsense" questions.
*   Manually write the *ideal* in-character refusal.
*   Add these pairs to your fine-tuning dataset (at least 100-200 examples).

## 2. Functional Grounding (JSON Output)

Collect data where the NPC performs an action alongside dialogue.

### Format
**Input:** `Player: "Draw your sword!"`
**Output:**
```json
{
  "text": "You want a fight? You got one!",
  "emotion": "Anger",
  "action": "Draw_Sword",
  "trust_change": -20
}
```

### Examples for Collection

| Trigger | Dialogue | Action | Emotion |
| :--- | :--- | :--- | :--- |
| **Greeting** | "Hail, traveler." | `Wave` | `Joy` |
| **Insult** | "Watch your tongue!" | `Play_Anim_Angry` | `Anger` |
| **Gift** | "For me? Thank you." | `Receive_Item` | `Surprise` |
| **Threat** | "Stand back!" | `Combat_Idle` | `Fear` |

## 3. Tools for Collection

1.  **Log Conversations:** Enable logging in `NPCInference.cpp` to save all player-NPC turns to a file.
2.  **Human Review:** Periodically review logs. If NPC answers a modern question, mark it for "Refusal Training".
3.  **Synthesis:** Use a larger LLM (GPT-4) to generate variations of refusals using your persona descriptions. "Generate 50 questions about modern tech and have a Medieval Knight answer them confusedly."

## 4. Immediate Action Items

*   [ ] Create `datasets/refusals.jsonl`
*   [ ] Add 50 examples of "I don't know" for spaceship/internet/gun queries.
*   [ ] Add 50 examples of "JSON Action" pairs (e.g., Attack -> Draw Sword).
