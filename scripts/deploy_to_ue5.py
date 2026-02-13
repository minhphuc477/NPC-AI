import os
import shutil
import sys
from pathlib import Path

def deploy():
    # Paths
    project_root = Path(__file__).parent.parent
    cpp_build_dir = project_root / "cpp" / "build" / "Release"
    cpp_lib_dir = project_root / "cpp" / "lib"
    
    # Check if build exists
    if not cpp_build_dir.exists():
        print(f"Error: Build directory {cpp_build_dir} does not exist. Run cmake --build first.")
        # Try Debug?
        cpp_build_dir_debug = project_root / "cpp" / "build" / "Debug"
        if cpp_build_dir_debug.exists():
             cpp_build_dir = cpp_build_dir_debug
             print(f"Found Debug build at {cpp_build_dir}")
        else:
             return

    # Create destination
    cpp_lib_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    libs = [
        "npc_inference.lib",
        # onnxruntime is in ../onnxruntime/lib
    ]
    
    # 1. Copy npc_inference.lib
    for lib in libs:
        src = cpp_build_dir / lib
        dst = cpp_lib_dir / lib
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
        else:
            print(f"Warning: {src} not found.")

    # 2. Copy ONNX Runtime
    # It might be in build/onnxruntime/lib
    ort_lib_dir = project_root / "cpp" / "build" / "onnxruntime" / "lib"
    ort_lib = ort_lib_dir / "onnxruntime.lib"
    if ort_lib.exists():
        shutil.copy2(ort_lib, cpp_lib_dir / "onnxruntime.lib")
        print(f"Copied {ort_lib} -> {cpp_lib_dir / 'onnxruntime.lib'}")
    else:
        print(f"Warning: ONNX Runtime lib not found at {ort_lib}")

    # 3. Copy headers?
    # NPCInference.Build.cs points to cpp/include, which already exists.
    
    print("Deployment complete.")

if __name__ == "__main__":
    deploy()
