import numpy as np
from pose_module import PoseModule
from sam2_module import SAM2Module


class AutoMasker:
    POSITIVE_REGIONS = {
            "upper": [PoseModule.LEFT_SHOULDER, PoseModule.RIGHT_SHOULDER, PoseModule.LEFT_ELBOW, PoseModule.RIGHT_ELBOW, PoseModule.LEFT_HIP, PoseModule.RIGHT_HIP],
            "lower": [PoseModule.LEFT_HIP, PoseModule.RIGHT_HIP, PoseModule.LEFT_KNEE, PoseModule.RIGHT_KNEE, PoseModule.LEFT_ANKLE, PoseModule.RIGHT_ANKLE],
            "shoes": [PoseModule.LEFT_ANKLE, PoseModule.RIGHT_ANKLE, PoseModule.LEFT_HEEL, PoseModule.RIGHT_HEEL, PoseModule.LEFT_FOOT_INDEX, PoseModule.RIGHT_FOOT_INDEX],
        }
    NEGATIVE_REGIONS = {
        "upper": [PoseModule.NOSE, PoseModule.LEFT_WRIST, PoseModule.RIGHT_WRIST, PoseModule.LEFT_KNEE, PoseModule.RIGHT_KNEE],
        "lower": [PoseModule.NOSE, PoseModule.LEFT_SHOULDER, PoseModule.RIGHT_SHOULDER, PoseModule.LEFT_WRIST, PoseModule.RIGHT_WRIST,
                   PoseModule.LEFT_HEEL, PoseModule.RIGHT_HEEL, PoseModule.LEFT_FOOT_INDEX, PoseModule.RIGHT_FOOT_INDEX],
        "shoes": [PoseModule.LEFT_KNEE, PoseModule.RIGHT_KNEE, PoseModule.LEFT_HIP, PoseModule.RIGHT_HIP]
    }

    def __init__(self, sam2: SAM2Module, pose: PoseModule):
        self.sam2 = sam2
        self.pose = pose

    def _to_pixel_coords(self, normalized_coords, image_shape):
        """Convert normalized (x, y) to pixel coordinates."""
        h, w = image_shape[:2]
        return [(x * w, y * h) for x, y in normalized_coords]

    def segment_region(self, image, region):
        """Segment a clothing region using pose landmarks as SAM2 point prompts."""
        pose_landmarks = self.pose.detect_pose(image)

        positive = self._to_pixel_coords(
            self.pose.get_landmark_coordinates(pose_landmarks, self.POSITIVE_REGIONS[region]), image.shape
        )
        negative = self._to_pixel_coords(
            self.pose.get_landmark_coordinates(pose_landmarks, self.NEGATIVE_REGIONS[region]), image.shape
        )

        self.sam2.set_image(image)
        return self.sam2.best_mask(positive, negative)
    


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    # Initialize modules
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_weights = "weights/sam2.1_hiera_large.pt"
    sam_module = SAM2Module(sam2_cfg, sam2_weights)
    pose_module = PoseModule("weights/pose_landmarker.task")

    auto_masker = AutoMasker(sam_module, pose_module)

    # Load image
    image = Image.open("test/person.png")
    image_np = np.array(image)

    def draw_mask_overlay(image, mask):
        """Utility to visualize mask overlay on image."""
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)  # Overlay mask with transparency
        plt.title("Segmented Region")
        plt.axis("off")
        plt.show()

    # Segment upper body
    mask = auto_masker.segment_region(image_np, "upper")
    draw_mask_overlay(image_np, mask)
    # Segment lower body
    mask = auto_masker.segment_region(image_np, "lower")
    draw_mask_overlay(image_np, mask)
    # Segment shoes
    mask = auto_masker.segment_region(image_np, "shoes")
    draw_mask_overlay(image_np, mask)