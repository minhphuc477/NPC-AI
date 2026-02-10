# C++ NPC Inference Implementation

C++ implementation of the NPC AI inference engine for Unreal Engine 5 integration.

## Quick Start

```powershell
# 1. Install dependencies
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools

# 2. Build project (auto-downloads ONNX Runtime and nlohmann/json)
cd cpp
cmake -B build -DDOWNLOAD_ONNXRUNTIME=ON
cmake --build build --config Release

# 3. Test
.\build\Release\test_inference.exe
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Features

- **Exact Prompt Compatibility**: Matches Python training format precisely
- **ONNX Runtime**: Optimized inference with CUDA support
- **Auto-Dependencies**: CMake automatically downloads ONNX Runtime and nlohmann/json
- **JSON CLI**: Compatible with existing Python `npc_cli.py` interface
- **UE5 Ready**: Integrates with `ue5/NPCDialogueClient.h`

## Project Structure

```
cpp/
├── include/           # Header files
│   ├── NPCInference.h      # Main inference API
│   ├── PromptFormatter.h   # Prompt formatting
│   └── ModelLoader.h       # ONNX model loading
├── src/              # Implementation files
│   ├── NPCInference.cpp
│   ├── PromptFormatter.cpp
│   ├── ModelLoader.cpp
│   └── main_cli.cpp        # Standalone CLI
├── tests/            # Unit tests
│   └── test_inference.cpp
├── CMakeLists.txt    # Build configuration
├── INSTALL.md        # Detailed installation guide
└── README.md         # This file
```

## Usage

### Standalone CLI (matches Python npc_cli.py)

```powershell
# Export model first
python ../scripts/export_to_onnx.py --export-tokenizer

# Run CLI
.\build\Release\npc_cli.exe ..\onnx_models\npc_model.onnx
```

### As Library

```cpp
#include "NPCInference.h"

NPCInference::NPCInferenceEngine engine;
engine.LoadModel("model.onnx");

std::string response = engine.GenerateFromContext(
    "You are a loyal guard",  // persona
    "Guard_1",                 // npc_id
    "Village gate",            // scenario
    "Hello, may I enter?"      // player_input
);
```

## Requirements

- CMake 3.15+
- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)
- (Auto-downloaded) ONNX Runtime 1.17+
- (Auto-downloaded) nlohmann/json 3.11+
- (Optional) CUDA Toolkit for GPU acceleration

## Integration with UE5

Three integration options:

1. **Process Spawn** - Spawn `npc_cli.exe` (easiest, uses existing `ue5/NPCDialogueClient.h`)
2. **Shared Library** - Load as DLL
3. **UE5 Plugin** - Native plugin (best performance)

See `../ue5/NPCDialogueClient.h` for the UE5 HTTP client interface.

## Documentation

- **[INSTALL.md](INSTALL.md)** - Detailed build and installation instructions
- **[../scripts/export_to_onnx.py](../scripts/export_to_onnx.py)** - Model export utility
- **[../ue5/NPCDialogueClient.h](../ue5/NPCDialogueClient.h)** - UE5 integration example
