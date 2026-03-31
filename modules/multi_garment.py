import numpy as np
from shared.schemas import PipelinePayload
from pipeline.tryon import try_on


def multi_garment_try_on(person_image: np.ndarray, garments: list) -> np.ndarray:
    """
    Apply multiple garments sequentially onto a person image.

    Args:
        person_image : np.ndarray — starting person photo
        garments     : list of dicts with "image" and "type" keys

    Returns:
        np.ndarray — final try-on image with all garments applied
    """
    payload = PipelinePayload(
        person_image  = person_image,
        garment_image = garments[0]["image"],
        garment_type  = garments[0]["type"],
    )

    for garment in garments:
        payload.garment_image = garment["image"]
        payload.garment_type  = garment["type"]
        payload.body_mask     = None   # reset so automasker generates correct region

        payload = try_on(payload)

        payload.person_image = payload.result_image

    return payload.result_image