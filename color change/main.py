"""
Cloth Color Changer Pipeline
-----------------------------
Stack:
  - SegFormer (mattmdjaga/segformer_b2_clothes) → clothing segmentation
  - SDXL Inpainting (diffusers/stable-diffusion-xl-1.0-inpainting-0.1) → recoloring

Install:
  pip install torch torchvision diffusers transformers accelerate pillow
  (add --index-url https://download.pytorch.org/whl/cu118 for CUDA)
"""

import torch
import numpy as np
import transformers
import diffusers
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid


# ──────────────────────────────────────────────
# CLOTHING LABELS (segformer_b2_clothes)
# ──────────────────────────────────────────────
LABEL_MAP = {
    0:  "Background",
    1:  "Hat",
    2:  "Hair",
    3:  "Sunglasses",
    4:  "Upper-clothes",   # shirts, jackets, tops
    5:  "Skirt",
    6:  "Pants",
    7:  "Dress",
    8:  "Belt",
    9:  "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

# Clothes-only labels (exclude body parts / accessories)
CLOTH_LABELS = {4, 5, 6, 7, 8, 17}  # adjust to your needs


# ──────────────────────────────────────────────
# 1. SEGMENTATION → MASK
# ──────────────────────────────────────────────
def load_segformer():
    print("Loading SegFormer clothing segmentation model...")
    processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model.eval()
    return processor, model


def get_cloth_mask(image: Image.Image, processor, model, target_labels=None) -> Image.Image:
    """
    Returns a binary PIL mask (white = cloth area, black = background).
    target_labels: set of label IDs to include. Defaults to all CLOTH_LABELS.
    """
    if target_labels is None:
        target_labels = CLOTH_LABELS

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits to original image size
    logits = outputs.logits  # (1, num_classes, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    seg_map = upsampled.argmax(dim=1).squeeze().numpy()  # (H, W)

    # Build binary mask
    mask_array = np.zeros_like(seg_map, dtype=np.uint8)
    for label_id in target_labels:
        mask_array[seg_map == label_id] = 255

    mask = Image.fromarray(mask_array).convert("RGB")
    return mask


# ──────────────────────────────────────────────
# 2. INPAINTING → RECOLOR
# ──────────────────────────────────────────────
def load_inpainting_pipeline(device="cuda"):
    print("Loading SDLX Inpainting pipeline...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload() if device == "cuda" else None
    return pipe


def recolor_cloth(
    image: Image.Image,
    mask: Image.Image,
    target_color: str,
    pipe,
    strength: float = 0.99,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: int = 42,
) -> Image.Image:
    """
    Recolors the masked cloth area to target_color.

    Args:
        image:          Original PIL image (RGB)
        mask:           Binary mask (white = area to change)
        target_color:   Natural language color, e.g. "navy blue", "crimson red"
        strength:       1.0 = full repaint, 0.5 = subtle shift
        seed:           For reproducibility
    """
    # SDXL inpainting expects 1024x1024
    orig_size = image.size
    image_1024 = image.resize((1024, 1024))
    mask_1024  = mask.resize((1024, 1024))

    prompt = (
        f"A cloth garment in {target_color} color, "
        "same fabric texture, same lighting, same wrinkles, photorealistic"
    )
    negative_prompt = (
        "different style, pattern change, low quality, blurry, "
        "watermark, extra limbs, deformed"
    )

    generator = torch.Generator().manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_1024,
        mask_image=mask_1024,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    # Resize back to original
    result = result.resize(orig_size)
    return result


# ──────────────────────────────────────────────
# 3. MAIN — Run the full pipeline
# ──────────────────────────────────────────────
def recolor(
    image_path: str,
    target_color: str,
    output_path: str = "output.png",
    target_labels: set = None,
    device: str = None,
):
    """
    Full pipeline: load image → segment cloth → inpaint new color → save.

    Args:
        image_path:    Path to input image
        target_color:  e.g. "bright red", "forest green", "pastel pink"
        output_path:   Where to save the result
        target_labels: Which clothing parts to recolor (default: all clothes)
        device:        "cuda" | "cpu" (auto-detected if None)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load image
    image = Image.open("/teamspace/uploads/download (2).jpg").convert("RGB")

    # Step 1: Segment
    seg_processor, seg_model = load_segformer()
    print("Generating cloth mask...")
    mask = get_cloth_mask(image, seg_processor, seg_model, target_labels)

    # Step 2: Inpaint
    pipe = load_inpainting_pipeline(device)
    print(f"Recoloring cloth to: {target_color}...")
    result = recolor_cloth(image, mask, target_color, pipe)

    # Save outputs
    result.save(output_path)
    mask.save(output_path.replace(".png", "_mask.png"))

    # Save comparison grid
    grid = make_image_grid([image, mask, result], rows=1, cols=3)
    grid.save(output_path.replace(".png", "_comparison.png"))

    print(f"Done! Saved to: {output_path}")
    return result, mask


# ──────────────────────────────────────────────
# 4. EXAMPLE USAGE
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Basic usage
    recolor(
        image_path="/teamspace/uploads/download (2).jpg",
        target_color="red",
        output_path="/teamspace/s3_folders/output_images/outputs.png",
        target_labels={4},
    )

    # Recolor only pants (label 6)
    # recolor(
    #     image_path="person.jpg",
    #     target_color="navy blue",
    #     output_path="output_pants.png",
    #     target_labels={6},
    # )

    # Recolor only upper clothes (label 4) to green
    # recolor(
    #     image_path="person.jpg",
    #     target_color="forest green",
    #     output_path="output_shirt.png",
    #     target_labels={4},
    # )png