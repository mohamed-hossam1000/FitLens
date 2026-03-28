import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Module:

    def __init__(self, model_cfg, checkpoint):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image):
        """Encode image features — call once per image before predicting."""

        self.predictor.set_image(image)

    def predict(self, positive_points, negative_points=None):
        """
        Predict masks from point prompts.
        Points are pixel coordinates as lists of (x, y) tuples.
        Returns (masks, scores) sorted best-first.
        """
        point_coords = positive_points
        point_labels = [1] * len(positive_points)  # 1 = positive

        if negative_points:
            point_coords = point_coords + negative_points
            point_labels = point_labels + [0] * len(negative_points)  # 0 = negative

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(point_coords),
            point_labels=np.array(point_labels),
            multimask_output=True,  # return 3 masks, pick best
        )

        # sort by confidence score
        sorted_indices = np.argsort(scores)[::-1]
        return masks[sorted_indices], scores[sorted_indices]

    def best_mask(self, positive_points, negative_points=None):
        """Convenience method — returns only the highest scoring mask."""

        masks, _ = self.predict(positive_points, negative_points)
        return masks[0]
    

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "weights/sam2.1_hiera_large.pt"
    sam_module = SAM2Module(model_cfg, checkpoint)

    image = Image.open("weights/person.png")
    image_np = np.array(image)
    sam_module.set_image(image_np)

    # Example: positive points on upper body, negative points on background
    positive_points = [(600, 800)]  # example coords
    # negative_points = [(50, 50), (200, 50), (50, 200)]     # example coords

    best_mask = sam_module.best_mask(positive_points)

    plt.imshow(best_mask)
    plt.title("Best Mask from SAM2")
    plt.show()