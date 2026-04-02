import os
import numpy as np
import cv2
from PIL import Image

from shared.schemas import PipelinePayload
from pipeline.loader import get_catvton_masker, _setup_catvton_path
from fitlens_utils.color import rgb_to_lab, lab_to_rgb, compute_dominant_lab, delta_e_mask, recolor
from config import WIDTH, HEIGHT


def recolor_garment(
    payload: PipelinePayload,
    de_threshold: float = 25,
) -> PipelinePayload:
    result_image = payload.result_image
    mask         = payload.body_mask

    if mask is None:
        _setup_catvton_path()
        from utils import resize_and_crop

        automasker = get_catvton_masker()
        pil_result = resize_and_crop(Image.fromarray(result_image), (WIDTH, HEIGHT))
        mask_pil   = automasker(pil_result, payload.garment_type)["mask"]
        mask       = np.array(mask_pil.convert("L")) > 127

    # resize mask to match result image if dimensions differ
    if mask.shape[:2] != result_image.shape[:2]:
        h, w = result_image.shape[:2]
        mask  = cv2.resize(
            mask.astype(np.uint8), (w, h),
            interpolation = cv2.INTER_NEAREST
        ).astype(bool)

    lab_img      = rgb_to_lab(result_image)
    dominant_lab = compute_dominant_lab(lab_img, mask)

    target_pixel = np.array([[payload.target_rgb]], dtype=np.uint8)
    target_lab   = rgb_to_lab(target_pixel)[0, 0]

    de_mask      = delta_e_mask(lab_img, dominant_lab, de_threshold)
    recolor_mask = (mask > 0) & de_mask

    recolored_lab        = recolor(lab_img, recolor_mask, dominant_lab, target_lab)
    payload.result_image = lab_to_rgb(recolored_lab)

    return payload