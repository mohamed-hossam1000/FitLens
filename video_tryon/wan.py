from diffusers import WanImageToVideoPipeline
from PIL import Image
import torch
from pathlib import Path
import numpy as np
import imageio

# ─────────────────────────────────────────────
# 🎬 CHOOSE YOUR PROMPT HERE
# ─────────────────────────────────────────────
# Options:
#   "walking"   - person walks forward naturally
#   "turning"   - 360 spin, great for dresses/abayas
#   "posing"    - slow model pose shifts, good for suits/formal
#   "windy"     - fabric blows in breeze, great for loose clothing

MOTION = "turning"  # ← change this

PROMPTS = {
    "walking": {
        "prompt": "a person walking forward, white background, natural fluid motion, realistic clothing movement, sharp outfit details, soft even studio lighting, full body shot, smooth motion, cinematic",
        "negative_prompt": "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, static, watermark, text, ugly, morphing, melting, jitter, background change, colored background, noisy background, extra clothing, floating fabric, color bleeding",
        "guidance_scale": 5.5,
    },
    "turning": {
        "prompt": "a person doing a slow 360 degree spin in place, white background, fabric flowing and draping naturally during rotation, full outfit visible, sharp clothing details, soft even studio lighting, full body shot, smooth rotation, cinematic",
        "negative_prompt": "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, static, watermark, text, ugly, morphing, melting, jitter, background change, colored background, noisy background, color bleeding, floating fabric, incomplete rotation",
        "guidance_scale": 5.5,
    },
    "posing": {
        "prompt": "a person shifting poses naturally, subtle confident movement, white background, slight weight shift from one leg to the other, natural hand movement, sharp outfit details, soft even studio lighting, full body shot, smooth motion, fashion editorial style, cinematic",
        "negative_prompt": "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, static, watermark, text, ugly, morphing, melting, jitter, background change, colored background, noisy background, walking, running, jumping, color bleeding",
        "guidance_scale": 5.5,
    },
    "windy": {
        "prompt": "a person standing with fabric and clothing gently blowing in a soft breeze, white background, natural wind effect on outfit, hair and fabric moving fluidly, sharp clothing details, soft even studio lighting, full body shot, smooth motion, cinematic",
        "negative_prompt": "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, static, watermark, text, ugly, morphing, melting, jitter, background change, colored background, noisy background, walking, running, color bleeding, violent wind",
        "guidance_scale": 5.5,
    },
}

selected = PROMPTS[MOTION]
print(f"🎬 Motion selected: '{MOTION}'")
print(f"   Prompt: {selected['prompt'][:80]}...")

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
print("\n=== Wan2.2 I2V-A14B - Optimized for A100 ===")

pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
print("✅ Model loaded successfully on A100.")

# ─────────────────────────────────────────────
# Image
# ─────────────────────────────────────────────
image_path = "/teamspace/studios/this_studio/tryon_result.png"
image = Image.open(image_path).convert("RGB")
image = image.resize((720, 480), Image.LANCZOS)
print(f"Image loaded: {image.size}")

# ─────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────
print("Generating video...")
output = pipe(
    image=image,
    prompt=selected["prompt"],
    negative_prompt=selected["negative_prompt"],
    num_frames=81,
    height=720,
    width=480,
    num_inference_steps=30,
    guidance_scale=selected["guidance_scale"],
    generator=torch.Generator("cuda").manual_seed(42),
)

video_frames = output.frames[0]

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
Path("output").mkdir(exist_ok=True)

# Convert to uint8 to suppress warnings
frames_np = np.stack([np.array(f) for f in video_frames])

# Fix: ensure uint8
if frames_np.dtype != np.uint8:
    frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
np.save("output/frames_backup.npy", frames_np)
print(f"✅ Frames backed up: {frames_np.shape}")

output_path = f"output/wan2.2_{MOTION}_a100.mp4"
writer = imageio.get_writer(output_path, format="ffmpeg", fps=24, quality=8, macro_block_size=1)
for frame in frames_np:
    writer.append_data(frame)
writer.close()
print(f"✅ Video saved: {output_path}")