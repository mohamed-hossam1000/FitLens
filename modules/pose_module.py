import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseModule:
    # MediaPipe Pose landmark indices
    NOSE             = 0
    LEFT_EYE_INNER   = 1
    LEFT_EYE         = 2
    LEFT_EYE_OUTER   = 3
    RIGHT_EYE_INNER  = 4
    RIGHT_EYE        = 5
    RIGHT_EYE_OUTER  = 6
    LEFT_EAR         = 7
    RIGHT_EAR        = 8
    MOUTH_LEFT       = 9
    MOUTH_RIGHT      = 10
    LEFT_SHOULDER    = 11
    RIGHT_SHOULDER   = 12
    LEFT_ELBOW       = 13
    RIGHT_ELBOW      = 14
    LEFT_WRIST       = 15
    RIGHT_WRIST      = 16
    LEFT_PINKY       = 17
    RIGHT_PINKY      = 18
    LEFT_INDEX       = 19
    RIGHT_INDEX      = 20
    LEFT_THUMB       = 21
    RIGHT_THUMB      = 22
    LEFT_HIP         = 23
    RIGHT_HIP        = 24
    LEFT_KNEE        = 25
    RIGHT_KNEE       = 26
    LEFT_ANKLE       = 27
    RIGHT_ANKLE      = 28
    LEFT_HEEL        = 29
    RIGHT_HEEL       = 30
    LEFT_FOOT_INDEX  = 31
    RIGHT_FOOT_INDEX = 32


    def __init__(self, model_path):
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.IMAGE,  # single image mode
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(self.options)

    def detect_pose(self, image):
        """Detect pose landmarks in an RGB numpy image. Returns landmarks for first person detected."""

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.pose_landmarker.detect(mp_image)
        return detection_result.pose_landmarks[0]
    
    def get_landmark_coordinates(self, pose_landmarks, indices, min_confidence=0.5):
        """Return normalized (x, y) for given indices, skipping low visibility/presence landmarks."""

        return [
            (pose_landmarks[i].x, pose_landmarks[i].y)
            for i in indices 
            if pose_landmarks[i].visibility >= min_confidence
            and pose_landmarks[i].presence >= min_confidence
        ]

if __name__ == "__main__":
    
    import numpy as np
    from mediapipe.tasks.python.vision import drawing_utils
    from mediapipe.tasks.python.vision import drawing_styles
    from mediapipe.tasks.python import vision
    from PIL import Image
    import matplotlib.pyplot as plt

    def draw_landmarks_on_image(rgb_image, detection_result):
        """Draw pose landmarks and connections onto a copy of the image."""

        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)

        return annotated_image


    model_path = "weights/pose_landmarker.task"
    pose_module = PoseModule(model_path)

    image = Image.open("test/person.png")
    image = np.array(image)
    pose_landmarks = pose_module.detect_pose(image)
    # convert normalized coords to pixel coords
    upper_body_landmarks = pose_module.get_region_landmarks(pose_landmarks, "upper") * np.array([image.shape[1], image.shape[0]])
    print("Upper body landmarks:", upper_body_landmarks)
    annotated_image = draw_landmarks_on_image(image, pose_module.pose_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image)))
    plt.imshow(annotated_image)
    plt.show()