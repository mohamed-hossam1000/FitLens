from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image
import torch, numpy as np, imageio
from pathlib import Path

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = load_image("/teamspace/studios/this_studio/tryon_result.png")

output = pipe(
    prompt="a person walking forward, white background, natural fluid motion, realistic clothing movement, sharp outfit details, soft even studio lighting, full body shot, smooth motion, cinematic",
    negative_prompt= "blurry, low quality, deformed, distorted limbs, extra limbs, flickering, static, watermark, text, ugly, morphing, melting, jitter, background change, colored background, noisy background, extra clothing, floating fabric, color bleeding",
    image=image,
    num_videos_per_prompt=1,
    num_frames=81,
    num_inference_steps=25,
    guidance_scale=5.5,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

Path("output").mkdir(exist_ok=True)
frames_np = np.stack([np.array(f) for f in output])
if frames_np.dtype != np.uint8:
    frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

writer = imageio.get_writer("output/cogvideox_tryon.mp4", format="ffmpeg", fps=24, quality=8, macro_block_size=1)
for frame in frames_np:
    writer.append_data(frame)
writer.close()
print("✅ Video saved: output/cogvideox_tryon.mp4")