"""
wan.py
Wan2.2 Image-to-Video generation module for FitLens.
Exposes a single callable: generate_video(image, motion, gender, output_path)
"""

from diffusers import WanImageToVideoPipeline
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import imageio
import cv2


# ── Motion Presets ─────────────────────────────────────────────────────────────
PROMPTS = {
    "male": {
        "walking": {
            "prompt": (
                "a male fashion model walking forward, white background, natural fluid motion, "
                "realistic clothing movement, sharp outfit details, soft even studio lighting, "
                "full body shot, smooth motion, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, extra clothing, floating fabric, color bleeding"
            ),
            "guidance_scale": 5.5,
        },
        "turning": {
            "prompt": (
                "a male fashion model doing a slow 360 degree spin in place, white background, "
                "fabric flowing and draping naturally during rotation, full outfit visible, "
                "sharp clothing details, soft even studio lighting, full body shot, smooth rotation, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, color bleeding, floating fabric, incomplete rotation"
            ),
            "guidance_scale": 5.5,
        },
        "posing": {
            "prompt": (
                "a male fashion model shifting poses naturally, subtle confident movement, white background, "
                "slight weight shift from one leg to the other, natural hand movement, sharp outfit details, "
                "soft even studio lighting, full body shot, smooth motion, fashion editorial style, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, walking, running, jumping, color bleeding"
            ),
            "guidance_scale": 5.5,
        },
        "windy": {
            "prompt": (
                "a male fashion model standing with fabric and clothing gently blowing in a soft breeze, "
                "white background, natural wind effect on outfit, hair and fabric moving fluidly, "
                "sharp clothing details, soft even studio lighting, full body shot, smooth motion, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, walking, running, color bleeding, violent wind"
            ),
            "guidance_scale": 5.5,
        },
    },
    "female": {
        "walking": {
            "prompt": (
                "a female fashion model walking forward elegantly, white background, natural fluid motion, "
                "fabric flowing gracefully, sharp outfit details, soft even studio lighting, "
                "full body shot, smooth motion, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, extra clothing, floating fabric, color bleeding"
            ),
            "guidance_scale": 5.5,
        },
        "turning": {
            "prompt": (
                "a female fashion model doing a slow elegant 360 degree spin in place, white background, "
                "fabric and dress flowing gracefully during rotation, full outfit visible, "
                "sharp clothing details, soft even studio lighting, full body shot, smooth rotation, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, color bleeding, floating fabric, incomplete rotation"
            ),
            "guidance_scale": 5.5,
        },
        "posing": {
            "prompt": (
                "a female fashion model shifting poses gracefully, elegant natural movement, white background, "
                "subtle weight shift, soft hand gestures, sharp outfit details, soft even studio lighting, "
                "full body shot, smooth motion, fashion editorial style, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, walking, running, jumping, color bleeding"
            ),
            "guidance_scale": 5.5,
        },
        "windy": {
            "prompt": (
                "a female fashion model standing with fabric and clothing gently blowing in a soft breeze, "
                "white background, natural wind effect on outfit, hair and dress flowing fluidly, "
                "sharp clothing details, soft even studio lighting, full body shot, smooth motion, cinematic"
            ),
            "negative_prompt": (
                "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, "
                "static, watermark, text, ugly, morphing, melting, jitter, background change, "
                "colored background, noisy background, walking, running, color bleeding, violent wind"
            ),
            "guidance_scale": 5.5,
        },
    },
}

VALID_MOTIONS = ["walking", "turning", "posing", "windy"]
VALID_GENDERS = ["male", "female"]

# ── Lazy-load the pipeline once ────────────────────────────────────────────────
_pipe = None

def _load_pipeline():
    global _pipe
    if _pipe is None:
        print("⏳ Loading Wan2.2 I2V pipeline...")
        _pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        _pipe = _pipe.to("cuda")
        _pipe.enable_attention_slicing()
        print("✅ Wan2.2 pipeline loaded.")
    return _pipe


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_video(
    image: np.ndarray,
    gender: str = "male",
    motion: str = "walking",
    output_path: str = "output/result.mp4",
    num_frames: int = 81,
    num_inference_steps: int = 30,
    seed: int = 42,
) -> str:
    """
    Generate a try-on animation video from a single result image.
    """
    if gender not in VALID_GENDERS:
        raise ValueError(f"Invalid gender '{gender}'. Choose from: {VALID_GENDERS}")
    if motion not in VALID_MOTIONS:
        raise ValueError(f"Invalid motion '{motion}'. Choose from: {VALID_MOTIONS}")

    print(f"🎬 Generating video — gender: '{gender}' | motion: '{motion}' | frames: {num_frames} | steps: {num_inference_steps}")

    # === Real Generation Path ===
    pipe   = _load_pipeline()
    preset = PROMPTS[gender][motion]

    pil_image = Image.fromarray(image).convert("RGB").resize((480, 720), Image.LANCZOS)

    print("⚡ Running Wan2.2 inference... (this may take several minutes)")

    output = pipe(
        image               = pil_image,
        prompt              = preset["prompt"],
        negative_prompt     = preset["negative_prompt"],
        num_frames          = num_frames,
        height              = 720,
        width               = 480,
        num_inference_steps = num_inference_steps,
        guidance_scale      = preset["guidance_scale"],
        generator           = torch.Generator("cuda").manual_seed(seed),
    )

    video_frames = output.frames[0]

    frames_np = np.stack([np.array(f) for f in video_frames])
    if frames_np.dtype != np.uint8:
        frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=24, codec="libx264")
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()

    print(f"✅ Video saved → {output_path}")
    return output_path