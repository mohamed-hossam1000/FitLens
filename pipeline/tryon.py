import os
import sys
import numpy as np
from PIL import Image

from shared.schemas import PipelinePayload
from pipeline.loader import get_catvton, get_catvton_masker, _setup_catvton_path, _restore_project_path
from config import WIDTH, HEIGHT


def try_on(payload: PipelinePayload) -> PipelinePayload:
    VTON_pipe, mask_processor, generator = get_catvton()
    automasker                           = get_catvton_masker()

    _setup_catvton_path()
    from utils import resize_and_crop, resize_and_padding
    _restore_project_path()          # ← restore immediately after import

    person_image  = resize_and_crop(Image.fromarray(payload.person_image),     (WIDTH, HEIGHT))
    garment_image = resize_and_padding(Image.fromarray(payload.garment_image), (WIDTH, HEIGHT))

    if payload.body_mask is None:
        mask = automasker(person_image, payload.garment_type)["mask"]
    else:
        mask = resize_and_crop(
            Image.fromarray(payload.body_mask.astype(np.uint8) * 255).convert("L"),
            (WIDTH, HEIGHT)
        )

    mask = mask_processor.blur(mask, blur_factor=9)

    payload.result_image = np.array(VTON_pipe(
        image               = person_image,
        condition_image     = garment_image,
        mask                = mask,
        num_inference_steps = 50,
        guidance_scale      = 3.5,
        generator           = generator,
    )[0])

    _restore_project_path()          # ← restore again after pipeline runs
    return payload