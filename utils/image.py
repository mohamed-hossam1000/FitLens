import numpy as np
from PIL import Image
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def save_image(image: np.ndarray, path: str):
    """Save an RGB numpy array to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    print(f"✅ Image saved → {path}")