import os
import torch

CATVTON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "CatVTON")

BASE_MODEL_PATH   = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "catvton")
ATTN_CKPT_VERSION = "mix"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
HEIGHT            = 1024
WIDTH             = 768

SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_WEIGHTS = "weights/sam2.1_hiera_large.pt"
POSE_WEIGHTS = "weights/pose_landmarker.task"

WAN_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_SEED     = 42




