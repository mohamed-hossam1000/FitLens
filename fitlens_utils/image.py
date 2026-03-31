import numpy as np
from PIL import Image
from pathlib import Path

# Project root — always absolute regardless of cwd
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as RGB numpy array. Resolves relative paths from project root."""
    full_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    return np.array(Image.open(full_path).convert("RGB"))


def save_image(image: np.ndarray, path: str):
    """Save an RGB numpy array to disk. Resolves relative paths from project root."""
    full_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(full_path)
    print(f"✅ Image saved → {full_path}")