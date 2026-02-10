# NPC Inference Plugin for Unreal Engine 5

Native integration of the NPC AI Inference Engine for Unreal Engine 5.

## Features

- **Native C++ Performance**: Runs directly in the engine process, no external server required.
- **Blueprint Support**: Easy-to-use Async nodes and Components.
- **No Python Runtime**: Uses ONNX Runtime and C++ directly.

## Installation

1. Copy this `NPCInference` folder into your project's `Plugins` directory.
   - If `Plugins` doesn't exist, create it at your project root: `YourProject/Plugins/NPCInference`.
2. Ensure you have the C++ Inference Engine built.
   - The plugin expects the `cpp` library to be available.
3. Regenerate Visual Studio project files (Right-click .uproject -> Generate Visual Studio project files).
4. Build your project in Visual Studio.
5. Enable the plugin in UE5 (Edit -> Plugins -> AI -> NPC Inference).

## Setup

1. **Model Data**: Copy your `tokenizer.model` and `npc_model.onnx` (if available) to `YourProject/Content/ModelData`.
2. **Initialization**:
   - In your GameInstance or Main Menu Level Blueprint:
   - Call `Get NPCInferenceSubsystem` -> `Initialize Engine`.
   - Pass the path to your Model Data (e.g., `Get Default Model Path`).

## Usage

### Add NPC to Actor
1. Open your NPC Character Blueprint.
2. Add the `NPCDialogue` component.
3. Configure `NPCID`, `Persona`, and `Scenario` in the Details panel.

### Generate Dialogue
1. Call `Request Response` on the `NPCDialogue` component.
2. Bind an event to the `On Response Generated` dispatcher to handle the result.

```blueprint
[Player Interacts] -> [Request Response (Input)] 
                                     |
[On Response Generated] -> [Show Dialogue UI (Response)]
```

## Troubleshooting

- **"Engine not ready"**: Ensure you called `Initialize Engine` at game start and the path is correct.
- **Link Errors**: Check `NPCInference.Build.cs` paths point correctly to the `cpp` folder.
