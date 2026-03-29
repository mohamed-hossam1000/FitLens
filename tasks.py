import numpy as np
from PIL import Image

from shared.schemas import PipelinePayload
from modules.CatVTON.utils import resize_and_crop, resize_and_padding
from config import auto_masker, VTON_pipe, mask_processor, generator, WIDTH, HEIGHT
from utils import rgb_to_lab, lab_to_rgb, compute_dominant_lab, delta_e_mask, recolor


def try_on(payload:PipelinePayload) -> PipelinePayload:
    """
    Recolor the main color of the garment in person_image to target_rgb.

    Args:
        payload: Pipeline payload carrying the person image garment_image and optional mask.

    Returns:
        The same payload with result_image of the person wearing the garment.
    """
    # ensure we have a segmentation mask
    if payload.body_mask == None:
            payload.body_mask = auto_masker.segment_region(payload.person_image, payload.garment_type)

    # preprocess inputs for VTON
    person_image  = resize_and_crop(Image.fromarray(payload.person_image),     (WIDTH, HEIGHT))
    mask = resize_and_crop(Image.fromarray(payload.body_mask.astype(np.uint8) * 255).convert("L"),     (WIDTH, HEIGHT))
    garment_image = resize_and_padding(Image.fromarray(payload.garment_image), (WIDTH, HEIGHT))

    # blur the binary mask to create a soft conditioning signal for VTON
    mask = mask_processor.blur(mask, blur_factor=9)

    # run the VTON pipeline and write the result to payload
    payload.result_image = np.array(VTON_pipe(
        image               = person_image,
        condition_image     = garment_image,
        mask                = mask,
        num_inference_steps = 50,
        guidance_scale      = 2.5,
        generator           = generator
    )[0])

    return payload



def recolor_garment(
    payload: PipelinePayload,
    de_threshold: float = 25,
) -> PipelinePayload:
    """
    Recolor the main color of the garment in person_image to target_rgb.

    Args:
        payload:      Pipeline payload carrying the person image and optional mask.
        target_rgb:   Desired color as an (R, G, B) tuple, e.g. (220, 50, 50).
        de_threshold: ΔE distance cutoff for main-color pixel selection.

    Returns:
        The same payload with result_image set to the recolored photo.
    """
    # ensure we have a segmentation mask
    if payload.body_mask is None:
        payload.body_mask = auto_masker.segment_region(
            payload.person_image, payload.garment_type
        )

    mask = payload.body_mask  # binary, shape: (H, W)

    # convert full image to L*a*b*
    lab_img = rgb_to_lab(payload.person_image)

    # find dominant color inside the masked region
    dominant_lab = compute_dominant_lab(lab_img, mask)

    # convert target RGB → L*a*b*
    target_pixel = np.array([[payload.target_rgb]], dtype=np.uint8)
    target_lab = rgb_to_lab(target_pixel)[0, 0]
    
    # build ΔE mask — main-color pixels inside the segment only
    de_mask = delta_e_mask(lab_img, dominant_lab, de_threshold)
    recolor_mask = (mask > 0) & de_mask

    # apply color shift, preserving L* (texture + shadows)
    recolored_lab = recolor(lab_img, recolor_mask, dominant_lab, target_lab)

    # convert back to RGB and write to payload
    payload.result_image = lab_to_rgb(recolored_lab)

    return payload
