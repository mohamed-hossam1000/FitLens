import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter

from modules.CatVTON.pipeline import CatVTONPipeline
from modules.CatVTON.utils import resize_and_crop, resize_and_padding
from diffusers.image_processor import VaeImageProcessor
from modules.automasker import AutoMasker
from modules.pose_module import PoseModule
from modules.sam2_module import SAM2Module


sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_weights = "weights/sam2.1_hiera_large.pt"
sam_module = SAM2Module(sam2_cfg, sam2_weights)
pose_module = PoseModule("weights/pose_landmarker.task")

auto_masker = AutoMasker(sam_module, pose_module)

BASE_MODEL_PATH   = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT_PATH    = "weights/catvton"
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

# Resize — same as their app.py
person_image  = resize_and_crop(person_image,     (WIDTH, HEIGHT))
mask = auto_masker.segment_region(np.array(person_image), "upper")
mask = resize_and_crop(Image.fromarray(mask.astype(np.uint8) * 255).convert("L"),     (WIDTH, HEIGHT))
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