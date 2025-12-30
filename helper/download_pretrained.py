'''
    File name: /helper/download_pretrained.py
    Author: Oliver Czerwinski
    Date created: 12/25/2025
    Date last modified: 12/27/2025
    Python Version: 3.9+
'''

from huggingface_hub import snapshot_download

repo_id = "Aconexx/RLBauernskat"
local_folder = "pretrained"

print(f"Downloading pretrained .safetensors files from {repo_id}...")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_folder,
    allow_patterns="*.safetensors"
)

print(f"Download complete. Files are stored in the '{local_folder}' folder.")