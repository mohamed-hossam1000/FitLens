import os
import sys
import torch
from diffusers.image_processor import VaeImageProcessor

from config import (
    BASE_MODEL_PATH, ATTN_CKPT_PATH, ATTN_CKPT_VERSION,
    DEVICE, SAM2_CFG, SAM2_WEIGHTS, POSE_WEIGHTS,
    WAN_MODEL_ID, CATVTON_ROOT,
)

_catvton_pipe   = None
_mask_processor = None
_generator      = None
_catvton_masker = None
_wan_pipe       = None


def _setup_catvton_path():
    os.chdir(CATVTON_ROOT)
    if CATVTON_ROOT not in sys.path:
        sys.path.insert(0, CATVTON_ROOT)

def _restore_project_path():
    project_root = os.path.dirname(CATVTON_ROOT)
    os.chdir(project_root)

def get_catvton():
    global _catvton_pipe, _mask_processor, _generator
    print("▶ get_catvton() called", flush=True)
    if _catvton_pipe is None:
        _setup_catvton_path()

        from model.pipeline import CatVTONPipeline

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


def get_catvton_masker():
    global _catvton_masker

    if _catvton_masker is None:
        _setup_catvton_path()

        from model.cloth_masker import AutoMasker

        print("⏳ Loading CatVTON AutoMasker...")
        _catvton_masker = AutoMasker(
            densepose_ckpt = os.path.join(ATTN_CKPT_PATH, "DensePose"),
            schp_ckpt      = os.path.join(ATTN_CKPT_PATH, "SCHP"),
            device         = DEVICE,
        )
        print("✅ CatVTON AutoMasker loaded.")

    return _catvton_masker


def get_sam2():
    """SAM2 — used only for GarmentSegmentor in extract scenario."""
    from modules.sam2_module import SAM2Module
    return SAM2Module(SAM2_CFG, SAM2_WEIGHTS)


def get_wan():
    global _wan_pipe

    if _wan_pipe is None:
        from diffusers import WanImageToVideoPipeline

        print("⏳ Loading Wan2.1 I2V pipeline...")
        _wan_pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_ID,
            torch_dtype       = torch.bfloat16,
            low_cpu_mem_usage = True,
        )
        _wan_pipe = _wan_pipe.to("cuda")
        _wan_pipe.enable_attention_slicing()
        print("✅ Wan2.1 loaded.")

    return _wan_pipe