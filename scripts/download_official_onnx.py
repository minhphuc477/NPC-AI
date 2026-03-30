from huggingface_hub import snapshot_download
import os

def download_official_onnx():
    repo_id = "microsoft/Phi-3-mini-4k-instruct-onnx"
    # The official repo organizes models by target. We want the CPU INT4 version for testing
    subfolder = "cpu_and_mobile/cpu-int4-rtn-block-32"
    output_dir = "models/phi3_onnx_official"
    
    print(f"Downloading official {repo_id} ({subfolder}) to {output_dir}...")
    
    # Download just the necessary files manually if the structure is complex
    # Or just use snapshot_download with the allow_patterns
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=output_dir,
    )
    print("Download complete.")

if __name__ == "__main__":
    download_official_onnx()
