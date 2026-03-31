import numpy as np
from PIL import Image

from shared.schemas import PipelinePayload
from modules.CatVTON.utils import resize_and_crop, resize_and_padding
from pipeline.loader import get_catvton, get_auto_masker
from config import WIDTH, HEIGHT


def try_on(payload: PipelinePayload) -> PipelinePayload:
    """
    Run CatVTON virtual try-on.

    Args:
        payload: PipelinePayload with person_image, garment_image, garment_type set.

    Returns:
        Same payload with result_image populated.
    """
    VTON_pipe, mask_processor, generator = get_catvton()
    auto_masker                          = get_auto_masker()

    if payload.body_mask is None:
        payload.body_mask = auto_masker.segment_region(
            payload.person_image, payload.garment_type
        )

    person_image  = resize_and_crop(Image.fromarray(payload.person_image), (WIDTH, HEIGHT))
    mask          = resize_and_crop(
        Image.fromarray(payload.body_mask.astype(np.uint8) * 255).convert("L"),
        (WIDTH, HEIGHT)
    )
    garment_image = resize_and_padding(Image.fromarray(payload.garment_image), (WIDTH, HEIGHT))

    mask = mask_processor.blur(mask, blur_factor=9)

    payload.result_image = np.array(VTON_pipe(
        image               = person_image,
        condition_image     = garment_image,
        mask                = mask,
        num_inference_steps = 50,
        guidance_scale      = 2.5,
        generator           = generator,
    )[0])

    return payload