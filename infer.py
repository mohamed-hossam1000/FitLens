import argparse
import os
import sys
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from torchvision.io import read_video, write_video

sys.path.insert(0, "models/catvton")

from modules.cloth_masker import AutoMasker
from modules.pipeline import V2TONPipeline


CATEGORY_MAP = {
    "upper_body": "upper",
    "lower_body": "lower",
    "dresses": "overall",
}


def load_video(path, normalize=True):
    video = read_video(path, pts_unit="sec", output_format="TCHW")[0]
    video = video.permute(1, 0, 2, 3).unsqueeze(0).float() / 255.0
    if normalize:
        video = video * 2 - 1
    return video  # (1, C, T, H, W)


def load_garment(path, height, width):
    tfm = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image = Image.open(path).convert("RGB")
    return tfm(image).unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)


def resize_video(tensor, height, width):
    b, c, t, h, w = tensor.shape
    tensor = tensor.reshape(b * t, c, h, w)
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.reshape(b, c, t, height, width)


def pad_frames(tensor):
    original_len = tensor.size(2)
    remainder = original_len % 4
    if remainder:
        last = tensor[:, :, -1:].repeat(1, 1, 4 - remainder, 1, 1)
        tensor = torch.cat([tensor, last], dim=2)
    return tensor, original_len


def repaint(person, mask, result):
    h = person.size(-1)
    k = h // 50 if (h // 50) % 2 != 0 else h // 50 + 1
    mask = rearrange(mask, "b c f h w -> (b f) c h w")
    mask = F.avg_pool2d(mask, k, stride=1, padding=k // 2)
    mask = rearrange(mask, "(b f) c h w -> b c f h w", b=person.size(0))
    return person * (1 - mask) + result * mask


def get_mask_and_pose(video_path, category, ckpt_path):
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(ckpt_path, "DensePose"),
        schp_ckpt=os.path.join(ckpt_path, "SCHP"),
        device="cuda",
    )
    result = automasker.process_video(
        mask_type=category,
        video_path=video_path,
        densepose_colormap=cv2.COLORMAP_VIRIDIS,
    )
    mask      = result["mask"].unsqueeze(0).float() / 255.0
    densepose = result["densepose"].unsqueeze(0).float() / 255.0 * 2 - 1
    return mask, densepose  # both (1, C, T, H, W)


def save_video(tensor, path, fps=24):
    video = (tensor[0] * 0.5 + 0.5).clamp(0, 1)
    video = (video.permute(1, 2, 3, 0).cpu() * 255).byte()  # (T, H, W, C)
    write_video(path, video, fps=fps)


def run_tryon(args, pipeline, video_path, garment_path, output_path):
    category = CATEGORY_MAP[args.category]

    person          = load_video(video_path)
    garment         = load_garment(garment_path, args.height, args.width)
    mask, densepose = get_mask_and_pose(video_path, category, args.catvton_ckpt_path)

    person    = resize_video(person,    args.height, args.width)
    mask      = resize_video(mask,      args.height, args.width)
    densepose = resize_video(densepose, args.height, args.width)

    person,    original_len = pad_frames(person)
    mask,      _            = pad_frames(mask)
    densepose, _            = pad_frames(densepose)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    result = pipeline.video_try_on(
        source_video=person,
        condition_image=garment,
        mask_video=mask,
        pose_video=densepose,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        slice_frames=args.slice_frames,
        pre_frames=args.pre_frames,
        generator=generator,
        use_adacn=True,
    )  # (B, T, H, W, C)

    result = result.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
    result = repaint(person, mask, result)
    result = result[:, :, :original_len]

    save_video(result, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",            type=str, required=True)
    parser.add_argument("--output_dir",           type=str, required=True)
    parser.add_argument("--base_model_path",      type=str, default="models/easyanimate_weights")
    parser.add_argument("--finetuned_model_path", type=str, default="models/catvton_weights/512-64K")
    parser.add_argument("--catvton_ckpt_path",    type=str, default="models/catvton_masker")
    parser.add_argument("--category",             type=str, default="upper_body", choices=CATEGORY_MAP.keys())
    parser.add_argument("--height",               type=int,   default=512)
    parser.add_argument("--width",                type=int,   default=384)
    parser.add_argument("--num_inference_steps",  type=int,   default=20)
    parser.add_argument("--guidance_scale",       type=float, default=3.0)
    parser.add_argument("--slice_frames",         type=int,   default=24)
    parser.add_argument("--pre_frames",           type=int,   default=8)
    parser.add_argument("--seed",                 type=int,   default=555)
    parser.add_argument("--mixed_precision",      type=str,   default="bf16", choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    pipeline = V2TONPipeline(
        base_model_path=args.base_model_path,
        finetuned_model_path=args.finetuned_model_path,
        load_pose=True,
        torch_dtype=dtype,
        device="cuda",
    )

    for pair in sorted(os.listdir(args.input_dir)):
        pair_dir     = os.path.join(args.input_dir, pair)
        video_path   = os.path.join(pair_dir, "video.mp4")
        garment_path = os.path.join(pair_dir, "garment.jpg")
        output_path  = os.path.join(args.output_dir, f"{pair}.mp4")

        if not os.path.isdir(pair_dir):
            continue
        if not os.path.exists(video_path) or not os.path.exists(garment_path):
            print(f"[SKIP] {pair} — missing video.mp4 or garment.jpg")
            continue
        if os.path.exists(output_path):
            print(f"[SKIP] {pair} — already done")
            continue

        print(f"[RUN] {pair}")
        run_tryon(args, pipeline, video_path, garment_path, output_path)
        print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()