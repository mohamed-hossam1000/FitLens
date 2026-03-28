"""
teammate_loader.py
──────────────────
Utility your teammate can import to load the segmented garment
produced by app.py into their virtual try-on pipeline.

Usage example (in your teammate's code):
    from teammate_loader import load_garment
    garment_rgba, garment_rgb, mask = load_garment("outputs/20240101_120000_cutout.png")
"""

import numpy as np
from PIL import Image
import os
import json
import glob


def load_garment(cutout_path: str):
    """
    Load a segmented garment cutout produced by app.py.

    Parameters
    ----------
    cutout_path : str
        Path to the *_cutout.png file.

    Returns
    -------
    garment_rgba : np.ndarray  shape (H, W, 4)  dtype uint8
        Full RGBA image — alpha channel = garment mask.
    garment_rgb  : np.ndarray  shape (H, W, 3)  dtype uint8
        RGB pixels of the garment only (background pixels are black).
    mask         : np.ndarray  shape (H, W)     dtype bool
        True where the garment is.
    """
    img  = Image.open(cutout_path).convert("RGBA")
    rgba = np.array(img)

    mask        = rgba[:, :, 3] > 0          # alpha > 0  →  garment pixel
    garment_rgb = rgba[:, :, :3].copy()
    garment_rgb[~mask] = 0                   # black out background

    return rgba, garment_rgb, mask


def load_garment_with_mask_file(cutout_path: str):
    """
    Same as load_garment() but also returns the raw .npy mask saved alongside.
    Useful if you prefer the boolean array over deriving it from alpha.
    """
    mask_path = cutout_path.replace("_cutout.png", "_mask.npy")
    rgba, garment_rgb, alpha_mask = load_garment(cutout_path)

    if os.path.exists(mask_path):
        npy_mask = np.load(mask_path)         # bool (H, W)
        return rgba, garment_rgb, npy_mask
    else:
        return rgba, garment_rgb, alpha_mask


def get_latest_output(output_dir: str = "outputs"):
    """
    Auto-picks the most recently exported cutout — handy during development.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "*_cutout.png")))
    if not files:
        raise FileNotFoundError(f"No cutout files found in '{output_dir}'")
    return files[-1]


def load_metadata(cutout_path: str) -> dict:
    """Load the JSON metadata saved alongside a cutout."""
    meta_path = cutout_path.replace("_cutout.png", "_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


# ── Quick sanity-check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    latest = get_latest_output()
    print(f"Loading: {latest}")
    rgba, rgb, mask = load_garment(latest)
    print(f"  RGBA shape : {rgba.shape}")
    print(f"  RGB shape  : {rgb.shape}")
    print(f"  Mask shape : {mask.shape}  |  garment pixels: {mask.sum()}")
    meta = load_metadata(latest)
    if meta:
        print(f"  Metadata   : {meta}")
