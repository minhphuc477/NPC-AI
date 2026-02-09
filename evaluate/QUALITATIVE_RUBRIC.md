# BD-NSCA Qualitative Evaluation Rubric

## Overview
This rubric provides a structured framework for human evaluation of NPC dialogue quality.
Each response is scored on a 1-5 scale across multiple dimensions.

---

## Dimensions

### 1. Persona Consistency (1-5)
How well does the response match the NPC's defined personality?

| Score | Description |
|-------|-------------|
| 5 | Perfect match - language, tone, and behavior exactly reflect persona |
| 4 | Strong match - minor deviations that don't break character |
| 3 | Moderate match - generally consistent but some off-character moments |
| 2 | Weak match - frequently breaks character or uses wrong tone |
| 1 | No match - completely ignores persona definition |

**Evaluation Points:**
- Speaking style (formal/informal, short/long sentences)
- Vocabulary appropriate to character role
- Emotional expression matches personality traits
- Knowledge scope matches character background

---

### 2. Context Awareness (1-5)
Does the response appropriately reflect the current game state?

| Score | Description |
|-------|-------------|
| 5 | Full awareness - response directly addresses all context elements |
| 4 | Good awareness - acknowledges most context factors |
| 3 | Partial awareness - some context reflected but gaps exist |
| 2 | Limited awareness - mostly ignores context |
| 1 | No awareness - response could apply to any situation |

**Context Elements to Check:**
- Behavior state (idle, alert, combat, etc.)
- Health status
- Mood/emotional state
- Nearby entities
- Time of day
- Location

---

### 3. Dialogue Naturalness (1-5)
How natural and fluent does the dialogue sound?

| Score | Description |
|-------|-------------|
| 5 | Perfectly natural - indistinguishable from human writing |
| 4 | Very natural - minor awkwardness barely noticeable |
| 3 | Moderately natural - some unnatural phrasings |
| 2 | Somewhat unnatural - frequently awkward or stilted |
| 1 | Unnatural - clearly machine-generated |

**Evaluation Points:**
- Grammar and syntax correctness
- Idiomatic expression usage
- Conversation flow
- Appropriate length (not too verbose)

---

### 4. Plot Relevance (1-5)
Does the response appropriately reference the narrative context?

| Score | Description |
|-------|-------------|
| 5 | Highly relevant - enriches the narrative |
| 4 | Relevant - connects appropriately to plot |
| 3 | Somewhat relevant - loose connection to story |
| 2 | Marginally relevant - mostly disconnected |
| 1 | Irrelevant - contradicts or ignores plot |

---

### 5. Response Appropriateness (1-5)
Is the response appropriate for the player's input?

| Score | Description |
|-------|-------------|
| 5 | Perfect response - exactly what a player would expect |
| 4 | Good response - appropriate with minor issues |
| 3 | Acceptable response - reasonable but could be better |
| 2 | Weak response - partially addresses input |
| 1 | Inappropriate - ignores or misunderstands input |

---

## Scoring Sheet Template

```
Sample ID: _______________
NPC Type: _______________
Scenario: _______________

| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| Persona Consistency | | |
| Context Awareness | | |
| Dialogue Naturalness | | |
| Plot Relevance | | |
| Response Appropriateness | | |

Total Score: ___ / 25
Average: ___ / 5

Overall Comments:
_________________________________
```

---

## Aggregation

For comparing models:
1. Evaluate at least 50 samples per model
2. Calculate average scores per dimension
3. Calculate overall average
4. Report standard deviation

**Minimum acceptable threshold:** Average >= 3.5/5 across all dimensions
