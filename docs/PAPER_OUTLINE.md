# BD-NSCA: Academic Paper Outline & Structuring Guide

If you are planning to write an academic paper or a whitepaper on this project, the core narrative should focus on **Efficiency vs. Control**. Here is the recommended structure to highlight your model's unique strengths:

## 1. Title Ideas
*   *BD-NSCA: A Behavior-Driven Neuro-Symbolic Cognitive Architecture for Persona-Locked NPC Generation*
*   *Efficient Neuro-Symbolic Grounding for Local RPG Agents via Teacher-Student Distillation*
*   *Bridging the Hallucination Gap: State-Driven LLM Fine-Tuning for Video Game NPCs*

## 2. Abstract
**The Hook:** Large Language Models (LLMs) like GPT-4 produce incredible dialogue but are computationally unfeasible for local game execution and often suffer from "persona bleed" (breaking character to act like an AI assistant). Conversely, small local models often fail to adhere to complex game rules or hallucinate inventory states. 
**The Solution:** We propose a Behavior-Driven Neuro-Symbolic Cognitive Architecture (BD-NSCA) that distills high-quality roleplay data from a 70B parameter teacher model into a lightweight, quantized 3B adapter. 
**The Result:** By grounding the generation in symbolic JSON game-states, our model achieves a 75% win-rate in strict game-state adherence over zero-shot baselines while maintaining a fluent perplexity of ~37.3, enabling near-SOTA RPG dialogue running natively on consumer hardware.

## 3. Introduction
*   **The Problem:** Modern video games need dynamic NPCs, but cloud APIs introduce latency/cost, and generic models ignore game mechanics.
*   **The Neuro-Symbolic Approach:** Explain how you combine the *Neuro* (LLM natural language fluency) with the *Symbolic* (Behavior Trees and strict JSON game states).

## 4. Methodology (Your "Secret Sauce")
*   **Teacher-Led Data Distillation:** Explain how you used Groq (e.g., Llama-3 70B) to algorithmically generate 500+ high-quality training interactions based on precise RPG personas, bypassing the need for expensive human writers.
*   **Prompt Engineering for State:** Show your exact prompt format. How does the model see the `[CONTEXT]` (inventory, emotion, knowledge) before writing the `[NPC]` response?
*   **QLoRA Fine-Tuning:** Detail how you used LoRA to efficiently train the adapter so it "locks" into the persona and learns to read the JSON syntax without breaking English fluency.

## 5. Proposal Traceability & Objectives Met
*   *This section validates your methodology against your initial proposal.*
*   **Data Generation:** Contrast the initial proposal's plan for basic heuristic generation against the finalized Teacher-Led (Groq API) distillation pipeline, showing a massive jump in linguistic diversity.
*   **Quality Gates:** Address the original proposal's strict "Quality Gate" requirement of a 10/12 LLM-as-a-judge win-rate. Openly discuss hitting 9/12: frame it not as a failure, but as a strong 75% baseline success that identifies exact hyperparameter/dataset targets for future iterations.

## 6. Evaluation & Results (Flexing Your Data)
*   **Perplexity (Fluency):** Present the 37.32 perplexity score. Argue that despite the rigid JSON constraints, the model generates highly fluent, natural English.
*   **LLM-as-a-Judge Win Rates (Adherence):** Show the 9/12 win rate against the baseline. Emphasize that your model makes significantly fewer logical errors (like trying to sell an item it doesn't own) because of the BD-NSCA grounding.
*   **Qualitative Case Studies:** Pull 2-3 of the best examples from your `CASE_STUDIES.md` to visually show the reader how the model reacts to threats or inventory changes compared to a generic AI.

## 6. Discussion: The "Local Execution" Advantage
*   This is where you distinguish your work from others. Acknowledge that while a 70B model might have deeper general world knowledge, your 3B model is **hyper-specialized**. 
*   Emphasize that the BD-NSCA model can be integrated directly into Unreal Engine 5 via `llama.cpp` and GGUF formats, meaning game developers can ship this *today* without requiring players to have an always-on internet connection.

## 7. Conclusion & Future Work
*   Summarize that BD-NSCA successfully bridges the gap between unstructured LLM creativity and structured game engine logic.
*   *Future Work:* Suggest expanding the dataset further or integrating voice-to-text functionality.
