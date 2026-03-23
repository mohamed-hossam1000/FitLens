import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter

# Path setup — works on any machine, no hardcoding
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Try-on/
_catvton_root = os.path.join(_project_root, "CatVTON")                       # Try-on/CatVTON/

sys.path.insert(0, _project_root)   # so we can do: from CatVTON.model.xxx
sys.path.insert(0, _catvton_root)   # so CatVTON internals can do: from model.xxx

from CatVTON.model.pipeline import CatVTONPipeline
from CatVTON.model.cloth_masker import AutoMasker
from diffusers.image_processor import VaeImageProcessor
from CatVTON.utils import init_weight_dtype, resize_and_crop, resize_and_padding

BASE_MODEL_PATH   = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT_PATH    = os.path.join(_catvton_root, "weights", "catvton")
ATTN_CKPT_VERSION = "mix"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
HEIGHT            = 1024
WIDTH             = 768

def load_model():
    os.chdir(_catvton_root)

    pipe = CatVTONPipeline(
        base_ckpt         = BASE_MODEL_PATH,
        attn_ckpt         = ATTN_CKPT_PATH,
        attn_ckpt_version = ATTN_CKPT_VERSION,
        weight_dtype      = torch.float16,
        use_tf32          = True,
        device            = DEVICE,
        skip_safety_check = True,
    )

    automasker = AutoMasker(
        densepose_ckpt = os.path.join(_catvton_root, "weights", "catvton", "DensePose"),
        schp_ckpt      = os.path.join(_catvton_root, "weights", "catvton", "SCHP"),
        device         = DEVICE,
    )

    mask_processor = VaeImageProcessor(
        vae_scale_factor = 8,
        do_normalize     = False,
        do_binarize      = True,
        do_convert_grayscale = True
    )

    print(f"CatVTON loaded on {DEVICE}")
    return pipe, automasker, mask_processor


def run_catvton(
    person_image_path,
    garment_image_path,
    cloth_type     = "overall",
    pipe           = None,
    automasker     = None,
    mask_processor = None,
    output_path    = None,
):
    # Default output path
    if output_path is None:
        output_path = os.path.join(_project_root, "tryon", "output", "tryon_result.png")

    # Load models if not passed in
    if pipe is None:
        pipe, automasker, mask_processor = load_model()

    # Load images
    person_image  = Image.open(person_image_path).convert("RGB")
    garment_image = Image.open(garment_image_path).convert("RGB")

    # Resize — same as their app.py
    person_image  = resize_and_crop(person_image,     (WIDTH, HEIGHT))
    garment_image = resize_and_padding(garment_image, (WIDTH, HEIGHT))

    # Generate mask automatically using AutoMasker
    mask = automasker(
        person_image,
        cloth_type
    )['mask']

    # Blur mask edges for smooth blending
    mask = mask_processor.blur(mask, blur_factor=9)

    # Run inference
    generator = torch.Generator(device=DEVICE).manual_seed(555)

    result_image = pipe(
        image               = person_image,
        condition_image     = garment_image,
        mask                = mask,
        num_inference_steps = 50,
        guidance_scale      = 3.5,
        generator           = generator,
    )[0]

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_image.save(output_path)
    print(f"Try-on image saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    pipe, automasker, mask_processor = load_model()

    result = run_catvton(
        person_image_path  = os.path.join(_project_root, "tryon", "test", "person.jpg"),
        garment_image_path = os.path.join(_project_root, "tryon", "test", "garment.jpg"),
        cloth_type         = "upper",
        pipe               = pipe,
        automasker         = automasker,
        mask_processor     = mask_processor,
        output_path        = os.path.join(_project_root, "tryon", "output", "tryon_result.png"),
    )
    print("Done:", result)
