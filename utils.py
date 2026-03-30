import numpy as np
import cv2


def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to float32 L*a*b*."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)


def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    """Convert a float32 L*a*b* image back to RGB uint8."""
    lab_u8 = np.clip(img_lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_u8, cv2.COLOR_Lab2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_dominant_lab(lab_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the mean L*a*b* of masked pixels — our dominant color."""
    pixels = lab_img[mask > 0]  # shape: (N, 3)
    return pixels.mean(axis=0)  # shape: (3,)


def delta_e_mask(lab_img: np.ndarray, dominant_lab: np.ndarray,
                 threshold: float) -> np.ndarray:
    """
    Build a boolean mask of pixels whose ΔE distance from dominant_lab
    is below the threshold — these are the main-color pixels to recolor.

    Uses simplified ΔE (Euclidean in L*a*b*), sufficient for this task.
    """
    diff = lab_img - dominant_lab           # broadcast over all pixels
    delta_e = np.linalg.norm(diff[:, :, 1:3], axis=2)  # per-pixel L2 distance
    return delta_e < threshold


def recolor(lab_img: np.ndarray, recolor_mask: np.ndarray,
            dominant_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
    """
    Shift a* and b* of masked pixels by the offset between dominant and target.
    L* is untouched so shadows and texture are preserved.
    """
    offset = target_lab - dominant_lab  # only a* and b* matter
    offset[0] = 0                       # zero out L* offset explicitly

    result = lab_img.copy()
    result[recolor_mask, 1] += offset[1]   # shift a*
    result[recolor_mask, 2] += offset[2]   # shift b*

    return np.clip(result, 0, 255)