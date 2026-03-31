"""
main.py — FitLens pipeline test runner.
Edit the CONFIG block, then run: python main.py
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIG ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO            = "single"       # "single" | "multi" | "extract"

PERSON_IMAGE_PATH   = "input/person.jpg"
FASHION_MODEL_PATH  = "/teamspace/studios/this_studio/FitLens/input/garment.jpg"
SINGLE_GARMENT_PATH = "/teamspace/studios/this_studio/FitLens/input/garment.jpg"
MULTI_GARMENT_PATHS = [
    "/teamspace/studios/this_studio/FitLens/input/aob-tshirt.jpg",
    "/teamspace/studios/this_studio/FitLens/input/A18AT-0YerL._AC_SY606_.jpg",
]

POSITIVE_POINTS  = [(300, 400)]      # SAM2 click on garment (extract scenario)
NEGATIVE_POINTS  = [(10, 10)]        # SAM2 click on background
EXTRACT_GARMENT_TYPE  = "upper"

TARGET_COLOR_RGB = (220, 50, 50)              # e.g. (220, 50, 50) or None to skip
MOTION           = None              # "walking"|"turning"|"posing"|"windy"|None

OUTPUT_IMAGE_PATH = "output/result.png"
OUTPUT_VIDEO_PATH = "output/result.mp4"

# ══════════════════════════════════════════════════════════════════════════════
# ── Imports ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from shared.schemas import PipelinePayload
from modules import preprocess_person, preprocess_garments, multi_garment_try_on, GarmentSegmentor
from pipeline import try_on, recolor_garment, generate_video
from pipeline.loader import get_sam2
from utils.image import load_image, save_image
from PIL import Image
import cv2


# ══════════════════════════════════════════════════════════════════════════════
# ── Scenarios ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_single():
    print("\n══ SCENARIO: Single Garment ══")
    person_image  = load_image(PERSON_IMAGE_PATH)
    garment_image = load_image(SINGLE_GARMENT_PATH)

    person_info  = preprocess_person(person_image)
    garment_info = preprocess_garments([garment_image])[0]

    print(f"   Gender:       {person_info['gender']}")
    print(f"   Garment type: {garment_info['type']}")

    payload = PipelinePayload(
        person_image  = person_image,
        garment_image = garment_info["image"],
        garment_type  = EXTRACT_GARMENT_TYPE,
        target_rgb    = TARGET_COLOR_RGB,
    )
    return try_on(payload), person_info["gender"]


def run_multi():
    print("\n══ SCENARIO: Multiple Garments ══")
    person_image   = load_image(PERSON_IMAGE_PATH)
    garment_images = [load_image(p) for p in MULTI_GARMENT_PATHS]

    person_info = preprocess_person(person_image)
    garments    = preprocess_garments(garment_images)

    print(f"   Gender:   {person_info['gender']}")
    print(f"   Garments: {[g['type'] for g in garments]}")

    final_image = multi_garment_try_on(person_image, garments)

    payload = PipelinePayload(
        person_image  = person_image,
        garment_image = garments[-1]["image"],
        garment_type  = garments[-1]["type"],
        target_rgb    = TARGET_COLOR_RGB,
        result_image  = final_image,
    )
    return payload, person_info["gender"]


def run_extract():
    print("\n══ SCENARIO: Extract Garment from Fashion Model ══")
    fashion_model_image = load_image(FASHION_MODEL_PATH)
    person_image        = load_image(PERSON_IMAGE_PATH)

    seg = GarmentSegmentor()
    seg.set_image(Image.fromarray(fashion_model_image))

    # in main.py these are hardcoded for testing
    # in Streamlit these come from user clicks
    for x, y in POSITIVE_POINTS:
        seg.add_click(x, y, is_positive=True)
    for x, y in NEGATIVE_POINTS:
        seg.add_click(x, y, is_positive=False)

    cutout = seg.get_cutout()  # RGBA PIL Image
    garment_image = np.array(cutout.convert("RGB"))  # convert for CLIP + try-on

    person_info  = preprocess_person(person_image)
    garment_info = preprocess_garments([garment_image])[0]

    print(f"   Gender:       {person_info['gender']}")
    print(f"   Garment type: {garment_info['type']}")

    payload = PipelinePayload(
        person_image  = person_image,
        garment_image = garment_info["image"],
        garment_type  = garment_info["type"],
        target_rgb    = TARGET_COLOR_RGB,
    )
    return try_on(payload), person_info["gender"]


# ══════════════════════════════════════════════════════════════════════════════
# ── Main ──────────────────────────────────────────────────────────────════════
# ══════════════════════════════════════════════════════════════════════════════

def main():
    runners = {"single": run_single, "multi": run_multi, "extract": run_extract}
    if SCENARIO not in runners:
        raise ValueError(f"Unknown SCENARIO '{SCENARIO}'. Choose: {list(runners)}")

    payload, gender = runners[SCENARIO]()
    result_image    = payload.result_image

    if TARGET_COLOR_RGB is not None:
        print(f"\n🎨 Recoloring garment → RGB{TARGET_COLOR_RGB}")
        payload.result_image = result_image
        payload      = recolor_garment(payload)
        result_image = payload.result_image

    save_image(result_image, OUTPUT_IMAGE_PATH)

    if MOTION is not None:
        generate_video(
            image        = result_image,
            gender       = gender,
            motion       = MOTION,
            output_path  = OUTPUT_VIDEO_PATH,
        )
    else:
        print("\nℹ️  Video skipped (MOTION = None).")

    print("\n🏁 Pipeline complete.")


if __name__ == "__main__":
    main()