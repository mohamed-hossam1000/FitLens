from huggingface_hub import snapshot_download
import os

# Path setup
_tryon_root = os.path.dirname(os.path.abspath(__file__))  # Try-on/tryon/

# Download CatVTON weights
repo_path = snapshot_download(
    repo_id   = "zhengchong/CatVTON",
    local_dir = os.path.join(_tryon_root, "weights", "catvton")
)

print("Downloaded to:", repo_path)
print("Contents:", os.listdir(os.path.join(_tryon_root, "weights", "catvton")))