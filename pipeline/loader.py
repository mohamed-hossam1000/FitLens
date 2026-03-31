"""
pipeline/loader.py
All heavy models are loaded lazily — only on first use.
Import get_*() functions wherever you need a model instance.
"""

import torch
from config import (
    BASE_MODEL_PATH, ATTN_CKPT_PATH, ATTN_CKPT_VERSION,
    DEVICE, SAM2_CFG, SAM2_WEIGHTS, POSE_WEIGHTS, WAN_MODEL_ID,
)

_catvton_pipe   = None
_mask_processor = None
_generator      = None
_auto_masker    = None
_wan_pipe       = None


def get_catvton():
    global _catvton_pipe, _mask_processor, _generator
    if _catvton_pipe is None:
        from diffusers.image_processor import VaeImageProcessor
        from modules.CatVTON.pipeline import CatVTONPipeline

        print("⏳ Loading CatVTON pipeline...")
        _catvton_pipe = CatVTONPipeline(
            base_ckpt         = BASE_MODEL_PATH,
            attn_ckpt         = ATTN_CKPT_PATH,
            attn_ckpt_version = ATTN_CKPT_VERSION,
            weight_dtype      = torch.float16,
            use_tf32          = True,
            device            = DEVICE,
            skip_safety_check = True,
        )
        _mask_processor = VaeImageProcessor(
            vae_scale_factor     = 8,
            do_normalize         = False,
            do_binarize          = True,
            do_convert_grayscale = True,
        )
        _generator = torch.Generator(device=DEVICE).manual_seed(555)
        print("✅ CatVTON loaded.")

    return _catvton_pipe, _mask_processor, _generator


def get_auto_masker():
    global _auto_masker
    if _auto_masker is None:
        from modules.sam2_module import SAM2Module
        from modules.pose_module import PoseModule
        from modules.automasker import AutoMasker

        print("⏳ Loading SAM2 + Pose + AutoMasker...")
        sam_module   = SAM2Module(SAM2_CFG, SAM2_WEIGHTS)
        pose_module  = PoseModule(POSE_WEIGHTS)
        _auto_masker = AutoMasker(sam_module, pose_module)
        print("✅ AutoMasker loaded.")

    return _auto_masker


def get_sam2():
    """Returns the SAM2Module directly — used for garment extraction scenario."""
    return get_auto_masker().sam2


def get_wan():
    global _wan_pipe
    if _wan_pipe is None:
        from diffusers import WanImageToVideoPipeline

        print("⏳ Loading Wan2.1 I2V pipeline...")
        _wan_pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_ID,
            torch_dtype      = torch.bfloat16,
            low_cpu_mem_usage = True,
        )
        _wan_pipe = _wan_pipe.to("cuda")
        _wan_pipe.enable_attention_slicing()
        print("✅ Wan2.1 loaded.")

    return _wan_pipe