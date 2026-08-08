# """Upload the contents of the `models/` folder to a Hugging Face model repo.
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export HF_REPO_ID="your-username/your-repo"
#   python upload_models.py
# """

import os
from huggingface_hub import login, upload_folder

# Read token and repo id from environment for safety
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "Pranjaldutta129/Vehicle-Damage-Models")

if not HF_TOKEN:
    raise SystemExit("Set HF_TOKEN environment variable first (export HF_TOKEN=hf_...)")

models_dir = os.path.join(os.path.dirname(__file__), "models")
if not os.path.isdir(models_dir):
    raise SystemExit(f"Models folder not found: {models_dir}")

login(token=HF_TOKEN)  # programmatic login

print(f"Uploading contents of {models_dir} to {HF_REPO_ID} ...")
upload_folder(folder_path=models_dir, repo_id=HF_REPO_ID, repo_type="model")
print("Upload complete.")