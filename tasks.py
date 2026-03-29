import numpy as np
from PIL import Image

from shared.schemas import PipelinePayload
from modules.CatVTON.utils import resize_and_crop, resize_and_padding
from config import auto_masker, VTON_pipe, mask_processor, generator, WIDTH, HEIGHT


def try_on_task(payload:PipelinePayload) -> PipelinePayload:

    if payload.body_mask == None:
            payload.body_mask = auto_masker.segment_region(payload.person_image, payload.garment_type)

    person_image  = resize_and_crop(Image.fromarray(payload.person_image),     (WIDTH, HEIGHT))
    mask = resize_and_crop(Image.fromarray(payload.body_mask.astype(np.uint8) * 255).convert("L"),     (WIDTH, HEIGHT))
    garment_image = resize_and_padding(Image.fromarray(payload.garment_image), (WIDTH, HEIGHT))

    mask = mask_processor.blur(mask, blur_factor=9)
    payload.result_image = np.array(VTON_pipe(
        image               = person_image,
        condition_image     = garment_image,
        mask                = mask,
        num_inference_steps = 50,
        guidance_scale      = 2.5,
        generator           = generator
    )[0])

    return payload