# preprocessing.py
# Central preprocessing module for FitLens.
# Handles:
#   - Gender detection (CLIP)
#   - Garment type detection (CLIP)
#   - Person segmentation + white background (SAM2)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import clip
import torch
import numpy as np
from PIL import Image, ImageOps


# ── CLIP Labels ────────────────────────────────────────────────────
GENDER_LABELS = [
    "a photo of a man or male person",
    "a photo of a woman or female person",
]

GARMENT_LABELS = [
    "a photo of a shirt or top or jacket or hoodie or blouse",
    "a photo of pants or jeans or skirt or shorts or trousers",
    "a photo of a dress or jumpsuit or overall outfit",
]

# ── Maps ───────────────────────────────────────────────────────────
GENDER_MAP = {
    0: "male",
    1: "female",
}

GARMENT_TYPE_MAP = {
    0: "upper",
    1: "lower",
    2: "overall",
}

# ── Load CLIP once at startup ──────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print(f"✅ CLIP loaded on {device}")


# ──────────────────────────────────────────────────────────────────
# Gender Detection
# ──────────────────────────────────────────────────────────────────

def detect_gender(person_image: np.ndarray) -> str:
    """
    Detect gender from person image using CLIP.

    Args:
        person_image : np.ndarray (RGB)

    Returns:
        "male" or "female"
    """
    image = clip_preprocess(
        Image.fromarray(person_image).convert("RGB")
    ).unsqueeze(0).to(device)

    text = clip.tokenize(GENDER_LABELS).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features  = clip_model.encode_text(text)
        similarity     = (image_features @ text_features.T).softmax(dim=-1)
        predicted      = similarity.argmax().item()
        confidence     = similarity[0][predicted].item()

    gender = GENDER_MAP[predicted]
    return gender


# ──────────────────────────────────────────────────────────────────
# Garment Type Detection
# ──────────────────────────────────────────────────────────────────

def detect_garment_type(garment_image: np.ndarray) -> str:
    """
    Detect garment type from garment image using CLIP.

    Args:
        garment_image : np.ndarray (RGB)

    Returns:
        "upper", "lower", or "overall"
    """
    image = clip_preprocess(
        Image.fromarray(garment_image).convert("RGB")
    ).unsqueeze(0).to(device)

    text = clip.tokenize(GARMENT_LABELS).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features  = clip_model.encode_text(text)
        similarity     = (image_features @ text_features.T).softmax(dim=-1)
        predicted      = similarity.argmax().item()
        confidence     = similarity[0][predicted].item()

    garment_type = GARMENT_TYPE_MAP[predicted]
    print(f"✅ Garment type detected: {garment_type} (confidence: {confidence:.2%})")
    return garment_type





# ──────────────────────────────────────────────────────────────────
# Full Person Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────

def preprocess_person(person_image: np.ndarray) -> dict:
    """
    Detect gender from person image.

    Args:
        person_image : np.ndarray (RGB)

    Returns:
        dict:
            - gender : str — "male" or "female"
    """
    gender = detect_gender(person_image)
    print(f"✅ Person preprocessed — gender: {gender}")
    return {
        "gender": gender,
    }

# ──────────────────────────────────────────────────────────────────
# Full Garment Preprocessing
# ──────────────────────────────────────────────────────────────────

def preprocess_garments(garment_images: list) -> list:
    """
    Auto-detect type for each garment image.

    Args:
        garment_images : list of np.ndarray (RGB)

    Returns:
        list of dicts:
            - image : np.ndarray
            - type  : str — "upper", "lower", or "overall"
    """

    garments = []
    for i, garment_image in enumerate(garment_images):
        garment_type = detect_garment_type(garment_image)
        garments.append({
            "image": garment_image,
            "type" : garment_type,
        })
        print(f"   Garment {i+1}: {garment_type}")
    return garments