from huggingface_hub import snapshot_download

model_name = "lmsys/vicuna-7b-v1.5"

local_dir = "./vicuna-7b-v1.5"

print(f"Downloading model: {model_name}")
print(f"Saving to:       {local_dir}")

snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("\n--- Download Complete! ---")
print(f"Your Vicuna model files are in: {local_dir}")