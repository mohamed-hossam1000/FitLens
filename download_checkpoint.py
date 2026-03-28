#!/usr/bin/env python3
"""
Run this once to download the SAM ViT-H checkpoint (~2.5 GB).
Usage:  python download_checkpoint.py
"""
import os, urllib.request, hashlib

URL      = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DEST_DIR = "checkpoints"
DEST     = os.path.join(DEST_DIR, "sam_vit_h_4b8939.pth")
MD5      = "4b8939a88964f0f4ff5f5b2642c598a6"

os.makedirs(DEST_DIR, exist_ok=True)

if os.path.exists(DEST):
    print(f"Checkpoint already exists at {DEST}")
else:
    print(f"Downloading SAM ViT-H checkpoint to {DEST} ...")
    urllib.request.urlretrieve(URL, DEST, reporthook=lambda b, bs, t:
        print(f"\r  {min(b*bs, t)/1e6:.1f} / {t/1e6:.1f} MB", end="", flush=True))
    print("\nDownload complete.")

# Verify
with open(DEST, "rb") as f:
    digest = hashlib.md5(f.read()).hexdigest()

if digest == MD5:
    print("✅ Checksum verified.")
else:
    print(f"⚠️  Checksum mismatch — expected {MD5}, got {digest}. Re-download.")
