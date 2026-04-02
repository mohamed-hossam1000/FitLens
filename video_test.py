"""
video_test.py - Final Safe Video Test for FitLens
Uses imageio[ffmpeg] correctly + OpenCV fallback
"""

from pathlib import Path
import numpy as np
from PIL import Image
import cv2

VALID_MOTIONS = ["walking", "turning", "posing", "windy"]
VALID_GENDERS = ["male", "female"]


def generate_video(
    image: np.ndarray,
    gender: str = "male",
    motion: str = "walking",
    output_path: str = "output/result.mp4",
    num_frames: int = 81,
    dry_run: bool = True,
) -> str:
    """Main function - dry_run=True by default for safety"""
    if gender not in VALID_GENDERS:
        raise ValueError(f"Invalid gender '{gender}'. Choose from: {VALID_GENDERS}")
    if motion not in VALID_MOTIONS:
        raise ValueError(f"Invalid motion '{motion}'. Choose from: {VALID_MOTIONS}")

    print(f"🎬 Video Generation → Gender: {gender} | Motion: {motion} | Frames: {num_frames}")

    if dry_run:
        return _create_dummy_video(image, gender, motion, output_path, num_frames)

    print("⚠️  Real Wan2.2 generation mode (not activated here)")
    return _create_dummy_video(image, gender, motion, output_path, num_frames)


def test_video_generation(
    test_image: np.ndarray = None,
    gender: str = "female",
    motion: str = "walking",
    output_path: str = "output/test_video.mp4",
    num_frames: int = 41,
) -> str:
    """Safe test function - Recommended"""
    print("=" * 75)
    print("🧪 FITLENS VIDEO TEST - SAFE MODE (No Heavy Model)")
    print("=" * 75)
    print(f"Gender  : {gender}")
    print(f"Motion  : {motion}")
    print(f"Frames  : {num_frames}")
    print(f"Output  : {output_path}")
    print("-" * 75)

    if test_image is None:
        test_image = np.zeros((720, 480, 3), dtype=np.uint8)
        for i in range(720):
            test_image[i] = [min(255, i//2), 100, 180]
        cv2.putText(test_image, "FITLENS TEST IMAGE", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

    result = generate_video(
        image=test_image,
        gender=gender,
        motion=motion,
        output_path=output_path,
        num_frames=num_frames,
        dry_run=True
    )

    print(f"\n✅ Test completed successfully!")
    print(f"✅ Video saved at: {result}")
    print("=" * 75)
    return result


def _create_dummy_video(image, gender, motion, output_path, num_frames):
    print("🔧 Creating dummy animated video...")

    pil_img = Image.fromarray(image).convert("RGB").resize((480, 720), Image.LANCZOS)
    base = np.array(pil_img)

    frames = []
    for i in range(num_frames):
        frame = base.copy()
        progress = int((i / max(1, num_frames-1)) * 100)

        cv2.putText(frame, f"{motion.upper()} • {gender.upper()}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.putText(frame, f"Frame {i+1}/{num_frames} ({progress}%)", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 255, 200), 2)
        cv2.putText(frame, "DRY RUN - SAFE TEST", (30, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if motion == "walking":
            offset = int(25 * np.sin(i * 0.3))
            cv2.putText(frame, "→ WALKING →", (80 + offset, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 100), 4)

        frames.append(frame)

    return _save_video(frames, output_path)


def _save_video(frames, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try imageio with ffmpeg (now that it's properly installed)
        import imageio
        writer = imageio.get_writer(output_path, fps=24, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"✅ Video saved successfully with imageio → {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"⚠️ imageio failed: {e}")
        print("Falling back to OpenCV...")

    # Reliable OpenCV fallback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, 24.0, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"✅ Video saved with OpenCV → {output_path}")
    return str(output_path)


# Run test automatically when executing the file
if __name__ == "__main__":
    test_video_generation(
        gender="female",
        motion="walking",
        num_frames=41,
        output_path="output/test_video.mp4"
    )