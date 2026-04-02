"""
main.py — FitLens pipeline test runner.
Edit the CONFIG block, then run: python main.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("▶ main.py started", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIG ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO            = "extract"       # "single" | "multi" | "extract"

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

TARGET_COLOR_RGB = None              # e.g. (220, 50, 50) or None to skip
MOTION           = None              # "walking"|"turning"|"posing"|"windy"|None

OUTPUT_IMAGE_PATH = "/teamspace/studios/this_studio/FitLens/output/result.png"
OUTPUT_VIDEO_PATH = "/teamspace/studios/this_studio/FitLens/output/result.mp4"
# ══════════════════════════════════════════════════════════════════════════════
# ── Imports ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from shared.schemas import PipelinePayload
from modules import preprocess_person, preprocess_garments, multi_garment_try_on, GarmentSegmentor
from pipeline import try_on, recolor_garment, generate_video
from fitlens_utils.image import load_image, save_image


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

def show_result(original: np.ndarray, result: np.ndarray):
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(result)
    axes[1].set_title("Try-On Result")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(str(output_dir / "result_preview.png"), dpi=150, bbox_inches="tight")
    print(f"✅ Preview saved → {output_dir / 'result_preview.png'}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ── Scenarios ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_single():
    print("\n══ SCENARIO: Single Garment ══")
    person_image  = load_image(PERSON_IMAGE_PATH)
    print("✅ Person image loaded")
    garment_image = load_image(SINGLE_GARMENT_PATH)
    print("✅ Garment image loaded")

    person_info  = preprocess_person(person_image)
    print(f"✅ Gender: {person_info['gender']}")
    garment_info = preprocess_garments([garment_image])[0]
    print(f"✅ Garment type: {garment_info['type']}")

    payload = PipelinePayload(
        person_image  = person_image,
        garment_image = garment_info["image"],
        garment_type  = EXTRACT_GARMENT_TYPE,
        target_rgb    = TARGET_COLOR_RGB,
    )
    print("✅ Payload created — starting try_on...")
    result = try_on(payload)
    print("✅ try_on complete")
    return result, person_info["gender"]


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

    for x, y in POSITIVE_POINTS:
        seg.add_click(x, y, is_positive=True)
    for x, y in NEGATIVE_POINTS:
        seg.add_click(x, y, is_positive=False)

    cutout        = seg.get_cutout()
    garment_image = np.array(cutout.convert("RGB"))

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

    if result_image is None:
        raise ValueError("Pipeline failed — result_image is None. Check try_on() output.")

    if TARGET_COLOR_RGB is not None:
        print(f"\n🎨 Recoloring garment → RGB{TARGET_COLOR_RGB}")
        payload = recolor_garment(payload)
        result_image = payload.result_image

    show_result(load_image(PERSON_IMAGE_PATH), result_image)
    save_image(result_image, OUTPUT_IMAGE_PATH)

    if MOTION is not None:
        generate_video(
            image       = result_image,
            gender      = gender,
            motion      = MOTION,
            output_path = OUTPUT_VIDEO_PATH,
        )
    else:
        print("\nℹ️  Video skipped (MOTION = None).")

    print("\n🏁 Pipeline complete.")


if __name__ == "__main__":
    main()