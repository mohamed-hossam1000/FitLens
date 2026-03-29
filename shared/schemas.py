from pydantic import BaseModel
import numpy as np
from typing import Optional, Literal, Tuple

class PipelinePayload(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    person_image: np.ndarray          # Original person photo (RGB)
    garment_image: np.ndarray         # Flat-lay garment photo (RGB)
    garment_type: Literal["upper", "lower", "shoes"]      # "upper" or "lower" or shoes
    target_rgb: Optional[Tuple[int, int, int]] = None # Desired color for recoloring, e.g. (220, 50, 50)
    body_mask: Optional[np.ndarray] = None        # Output of seg_module: binary mask of region to replace
    result_image: Optional[np.ndarray] = None     # Output of tryon_model: try-on composite
    video_frames: Optional[list[np.ndarray]] = None  # Output of video_module: 25 animation frames