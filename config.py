import torch
from diffusers.image_processor import VaeImageProcessor

from modules.CatVTON.pipeline import CatVTONPipeline
from modules.automasker import AutoMasker
from modules.pose_module import PoseModule
from modules.sam2_module import SAM2Module

sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_weights = "weights/sam2.1_hiera_large.pt"
sam_module = SAM2Module(sam2_cfg, sam2_weights)

pose_weights = "weights/pose_landmarker.task"
pose_module = PoseModule(pose_weights)

auto_masker = AutoMasker(sam_module, pose_module)

BASE_MODEL_PATH   = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT_PATH    = "weights/catvton"
ATTN_CKPT_VERSION = "mix"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
HEIGHT            = 1024
WIDTH             = 768

VTON_pipe = CatVTONPipeline(
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

generator = torch.Generator(device=DEVICE).manual_seed(555)