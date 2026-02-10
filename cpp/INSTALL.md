# C++ Installation and Build Guide

## Quick Start

### Automated Installation (Windows)

```powershell
# Install CMake
winget install Kitware.CMake

# Install Visual Studio Build Tools with C++ support
winget install Microsoft.VisualStudio.2022.BuildTools

# Navigate to cpp directory and build
cd cpp
cmake -B build -DDOWNLOAD_ONNXRUNTIME=ON
cmake --build build --config Release
```

The CMake configuration will automatically download:
- ONNX Runtime (CPU or GPU version)
- nlohmann/json for JSON parsing

---

## Detailed Instructions

### Prerequisites

**Required:**
1. **CMake 3.15+**
   - Download: https://cmake.org/download/
   - Or: `winget install Kitware.CMake`

2. **C++ Compiler** (Visual Studio 2019+ on Windows)
   - Full IDE: https://visualstudio.microsoft.com/downloads/
   - Build Tools only: `winget install Microsoft.VisualStudio.2022.BuildTools`
   - **Important**: Select "Desktop development with C++" workload

**Optional:**
- CUDA Toolkit (for GPU acceleration)

### Build Steps

**Step 1: Configure Project**
```powershell
cd cpp
cmake -B build -DDOWNLOAD_ONNXRUNTIME=ON
```

Options:
- `-DUSE_CUDA=OFF` - Disable CUDA support (default: ON)
- `-DBUILD_TESTS=OFF` - Skip building tests (default: ON)
- `-DDOWNLOAD_ONNXRUNTIME=OFF` - Use manual ONNX Runtime installation

**Step 2: Build**
```powershell
cmake --build build --config Release
```

**Step 3: Test**
```powershell
# Run unit tests
.\build\Release\test_inference.exe

# Test CLI (after exporting model)
.\build\Release\npc_cli.exe ..\..\onnx_models\npc_model.onnx
```

---

## Manual ONNX Runtime Installation

If automatic download fails:

```powershell
# Run the manual installation script
..\scripts\install_onnxruntime.ps1

# Then build with manual paths
cmake -B build -DDOWNLOAD_ONNXRUNTIME=OFF
cmake --build build --config Release
```

---

## Exporting Model to ONNX

Before using the C++ CLI, export your PyTorch model:

```powershell
# From project root
python scripts/export_to_onnx.py --export-tokenizer
```

This creates:
- `onnx_models/npc_model.onnx` - Model file
- `onnx_models/tokenizer/` - Tokenizer files

---

## Usage

### As Standalone CLI

```powershell
# Start the CLI
.\build\Release\npc_cli.exe path\to\model.onnx

# Send JSON requests via stdin
echo '{"context": {"npc_id": "Guard_1", "persona": "You are a guard", "scenario": "Gate"}, "player_input": "Hello"}' | .\build\Release\npc_cli.exe model.onnx
```

### As Library

```cpp
#include "NPCInference.h"

// Create engine
NPCInference::NPCInferenceEngine engine;
engine.LoadModel("path/to/model.onnx");

// Generate response
std::string response = engine.GenerateFromContext(
    "You are a guard",  // persona
    "Guard_1",          // npc_id  
    "Gate",             // scenario
    "Hello"             // player_input
);
```

---

## Troubleshooting

**CMake not found after installation**
- Restart your terminal
- Add CMake to PATH manually: `C:\Program Files\CMake\bin`

**ONNX Runtime download fails**
- Use manual install: `..\scripts\install_onnxruntime.ps1`
- Or download manually from: https://github.com/microsoft/onnxruntime/releases

**Build errors about missing headers**
- Ensure Visual Studio C++ tools are installed
- Run from "Developer Command Prompt for VS"

**CUDA errors**
- Disable CUDA: `cmake -B build -DUSE_CUDA=OFF`
- Or install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

---

## Next Steps

1. Export model: `python scripts/export_to_onnx.py --export-tokenizer`
2. Test C++ build: `.\build\Release\test_inference.exe`
3. Integrate with UE5 using existing `ue5/NPCDialogueClient.h`
