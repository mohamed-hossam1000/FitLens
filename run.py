import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter

from modules.CatVTON.pipeline import CatVTONPipeline
from modules.CatVTON.utils import resize_and_crop, resize_and_padding
from diffusers.image_processor import VaeImageProcessor

# Path setup — works on any machine, no hardcoding
_catvton_root = "/teamspace/studios/this_studio/FitLens"                       # Try-on/CatVTON/

sys.path.insert(0, _catvton_root)   # so CatVTON internals can do: from model.xxx

BASE_MODEL_PATH   = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT_PATH    = os.path.join(_catvton_root, "weights", "catvton")
ATTN_CKPT_VERSION = "mix"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
HEIGHT            = 1024
WIDTH             = 768



pipe = CatVTONPipeline(
    base_ckpt         = BASE_MODEL_PATH,
    attn_ckpt         = ATTN_CKPT_PATH,
    attn_ckpt_version = ATTN_CKPT_VERSION,
    weight_dtype      = torch.float16,
    use_tf32          = True,
    device            = DEVICE,
    skip_safety_check = True,
)

mask_processor = VaeImageProcessor(
        vae_scale_factor = 8,
        do_normalize     = False,
        do_binarize      = True,
        do_convert_grayscale = True
    )

print(f"CatVTON loaded on {DEVICE}")

person_image  = Image.open("test/person.jpg").convert("RGB")
garment_image = Image.open("test/garment2.jpg").convert("RGB")
mask          = Image.open("test/mask.png").convert("L")

# Resize — same as their app.py
person_image  = resize_and_crop(person_image,     (WIDTH, HEIGHT))
mask  = resize_and_crop(mask,     (WIDTH, HEIGHT))
garment_image = resize_and_padding(garment_image, (WIDTH, HEIGHT))

mask = mask_processor.blur(mask, blur_factor=9)

# Run inference
generator = torch.Generator(device=DEVICE).manual_seed(555)

result_image = pipe(
    image               = person_image,
    condition_image     = garment_image,
    mask                = mask,
    num_inference_steps = 50,
    guidance_scale      = 2.5,
    generator           = generator,
)[0]

# Save output
result_image.save("test/tryon_result.png")
print(f"Try-on image saved to: 'test/tryon_result.png'")