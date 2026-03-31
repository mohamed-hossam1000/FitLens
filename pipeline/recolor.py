import numpy as np

from shared.schemas import PipelinePayload
from pipeline.loader import get_auto_masker
from utils.color import rgb_to_lab, lab_to_rgb, compute_dominant_lab, delta_e_mask, recolor
import cv2


def recolor_garment(
    payload: PipelinePayload,
    de_threshold: float = 25,
) -> PipelinePayload:
    """
    Recolor the dominant color of the garment in result_image to target_rgb.

    Args:
        payload:      PipelinePayload with result_image, garment_type, target_rgb set.
        de_threshold: ΔE cutoff for main-color pixel selection.

    Returns:
        Same payload with result_image updated to recolored photo.
    """
    auto_masker = get_auto_masker()

    if payload.body_mask is None:
        payload.body_mask = auto_masker.segment_region(
            payload.result_image, payload.garment_type
        )

    mask    = payload.body_mask
    result_image = payload.result_image
    
    if mask.shape[:2] != result_image.shape[:2]:
        h, w = result_image.shape[:2]
        mask  = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation = cv2.INTER_NEAREST
        ).astype(bool)
    lab_img = rgb_to_lab(payload.result_image)
    dominant_lab = compute_dominant_lab(lab_img, mask)

    target_pixel = np.array([[payload.target_rgb]], dtype=np.uint8)
    target_lab   = rgb_to_lab(target_pixel)[0, 0]

    de_mask      = delta_e_mask(lab_img, dominant_lab, de_threshold)
    recolor_mask = (mask > 0) & de_mask

    recolored_lab        = recolor(lab_img, recolor_mask, dominant_lab, target_lab)
    payload.result_image = lab_to_rgb(recolored_lab)

    return payload