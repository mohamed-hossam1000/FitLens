import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE      = "vit_h"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


class GarmentSegmentor:
    """
    SAM-based garment segmentation.
    No UI dependency — import this class into any frontend or pipeline.

    Typical usage
    -------------
    seg = GarmentSegmentor()
    seg.set_image(pil_image)
    seg.add_click(x=320, y=150, is_positive=True)
    cutout = seg.get_cutout()       # RGBA PIL Image  → pass to try-on model
    mask   = seg.get_mask()         # bool np.ndarray → (H, W)
    """

    def __init__(self, checkpoint: str = CHECKPOINT_PATH,
                 model_type: str = MODEL_TYPE,
                 device: str = DEVICE):
        print(f"Loading SAM ({model_type}) on {device} ...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        print("SAM loaded ✓")
        self._reset_state()

    # Stat

    def _reset_state(self):
        self.image_np     = None
        self.points       = []          # list of [x, y]
        self.labels       = []          # list of 0 / 1
        self.all_masks    = None        # (3, H, W) — SAM's 3 candidates
        self.all_scores   = None        # (3,)
        self.current_mask = None        # (H, W) bool — active selection
        self.mask_index   = 0           # which of the 3 candidates is active

    #  Image loading 

    def set_image(self, image: Image.Image):
        """
        Call once per image before any clicks.
        Accepts a PIL Image (any mode — converted to RGB internally).
        """
        self._reset_state()
        self.image_np = np.array(image.convert("RGB"))
        self.predictor.set_image(self.image_np)

    #  Clicking 

    def add_click(self, x: int, y: int,
                  is_positive: bool = True,
                  mask_index: int = 0):
        """
        Add a click and recompute the mask.

        Parameters
        ----------
        x, y         : pixel coordinates (integers)
        is_positive  : True  = include this region (green)
                       False = exclude this region  (red)
        mask_index   : 0 = tightest / garment-only  (recommended default)
                       1 = medium
                       2 = whole body / loose
        Returns
        -------
        all_masks   : np.ndarray (3, H, W) — all three SAM candidates
        all_scores  : np.ndarray (3,)      — confidence per candidate
        """
        if self.image_np is None:
            raise RuntimeError("Call set_image() before add_click().")

        self.points.append([x, y])
        self.labels.append(1 if is_positive else 0)
        self.mask_index = mask_index

        self.all_masks, self.all_scores, _ = self.predictor.predict(
            point_coords     = np.array(self.points),
            point_labels     = np.array(self.labels),
            multimask_output = True,
        )
        self.current_mask = self.all_masks[mask_index]
        return self.all_masks, self.all_scores

    def select_mask(self, mask_index: int):
        """
        Switch between the 3 SAM mask candidates without adding a new click.
        Call this when the user adjusts the mask-size slider.

        mask_index : 0 = tightest, 1 = medium, 2 = loosest
        """
        if self.all_masks is None:
            raise RuntimeError("No masks available — add at least one click first.")
        if mask_index not in (0, 1, 2):
            raise ValueError("mask_index must be 0, 1, or 2.")
        self.mask_index   = mask_index
        self.current_mask = self.all_masks[mask_index]
        return self.current_mask

    def undo(self):
        """
        Remove the last click and recompute.
        Returns (all_masks, all_scores) or (None, None) if no clicks remain.
        """
        if not self.points:
            return None, None

        self.points.pop()
        self.labels.pop()

        if not self.points:
            self.all_masks = self.all_scores = self.current_mask = None
            return None, None

        self.all_masks, self.all_scores, _ = self.predictor.predict(
            point_coords     = np.array(self.points),
            point_labels     = np.array(self.labels),
            multimask_output = True,
        )
        self.current_mask = self.all_masks[self.mask_index]
        return self.all_masks, self.all_scores

    def reset(self):
        """Clear all clicks. The loaded image stays ready for new clicks."""
        self.points       = []
        self.labels       = []
        self.all_masks    = self.all_scores = self.current_mask = None
        self.mask_index   = 0

    # ── Outputs 

    def get_cutout(self) -> Image.Image:
        """
        Primary output for the try-on pipeline.
        Returns an RGBA PIL Image:
          - Opaque pixels  = garment
          - Transparent pixels = background
        Save with:  cutout.save("garment.png")   (PNG preserves transparency)
        """
        if self.current_mask is None:
            raise RuntimeError("No mask — add at least one click first.")
        rgba = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (self.current_mask * 255).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    def get_mask(self) -> np.ndarray:
        """
        Returns the binary mask as a boolean numpy array of shape (H, W).
        True  = garment pixel
        False = background pixel
        """
        if self.current_mask is None:
            raise RuntimeError("No mask — add at least one click first.")
        return self.current_mask.astype(bool)

    def get_all_masks(self):
        """
        Returns all three SAM candidate masks and their confidence scores.
        Useful if the frontend wants to let the user pick visually.

        Returns
        -------
        all_masks  : np.ndarray (3, H, W) or None
        all_scores : np.ndarray (3,)      or None
        """
        return self.all_masks, self.all_scores

    def get_preview(self) -> np.ndarray:
        """
        Returns an RGB numpy array (H, W, 3) with the mask overlaid in green
        and click points drawn as colored circles.
        Useful for displaying progress in any frontend.
        """
        if self.image_np is None:
            raise RuntimeError("No image loaded.")
        if self.current_mask is None:
            return self.image_np.copy()

        overlay = self.image_np.copy()
        colored = np.zeros_like(self.image_np)
        colored[self.current_mask == 1] = [0, 200, 120]
        overlay = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)

        contour_mask = (self.current_mask * 255).astype(np.uint8)
        contours, _  = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 230, 120), 2)

        for (x, y), lbl in zip(self.points, self.labels):
            color = (80, 220, 120) if lbl == 1 else (220, 80, 80)
            cv2.circle(overlay, (int(x), int(y)), 8, color, -1)
            cv2.circle(overlay, (int(x), int(y)), 8, (255, 255, 255), 2)

        return overlay