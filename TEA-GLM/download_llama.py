from huggingface_hub import snapshot_download, HfApi, login

# This is the Llama 2 CHAT model, which matches the [INST]
# template in your conversation.py file.
model_name = "meta-llama/Llama-2-7b-chat-hf"

# We'll save it to a new directory
local_dir = "./llama-2-7b-chat-hf"

print("Attempting to log in to Hugging Face...")
try:
    api = HfApi()
    api.whoami()
    print("Login successful (token already found).")
except Exception:
    print("Could not find Hugging Face token. Please log in.")
    login()

print(f"\nDownloading model: {model_name}")
print(f"Saving to:       {local_dir}")
print("This may take a while (~14GB)...")

snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=True  # Use the logged-in token
)

print("\n--- Download Complete! ---")
print(f"Your Llama 2 Chat model files are in: {local_dir}")