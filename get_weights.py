import os
import urllib.request

os.makedirs("weights", exist_ok=True)

weights = {
    "weights/pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "weights/sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

for path, url in weights.items():
    if os.path.exists(path):
        print(f"Already exists: {path}")
        continue
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved: {path}")