import mediapipe as mp
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Define the exact structural skeletal pairings for drawing manual lines
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),               # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),               # Index
    (5, 9), (9, 10), (10, 11), (11, 12),          # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),        # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

# ---------------------------------------------------------------------------
# Defaults — edit these if your machine layout differs from this repo.
# Model asset lives next to this module in supplements/.
# ---------------------------------------------------------------------------
SUPPLEMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SUPPLEMENTS_DIR / "hand_landmarker.task")


def mediapipe_label_for_side(side: str, *, selfie: bool = False) -> str:
    """MediaPipe handedness label to match after webcam horizontal flip."""
    robot_hand = "Left" if side == "left" else "Right"
    if selfie:
        return robot_hand
    return "Right" if robot_hand == "Left" else "Left"


@dataclass
class HandResult:
    joint_pos: np.ndarray
    keypoint_2d: Any
    wrist_rot: np.ndarray


class HandDetector:
    """MediaPipe hand detector supporting one or more hands per frame."""

    def __init__(
        self,
        num_hands: int = 1,
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.8,
        model_asset_path: str = DEFAULT_MODEL_PATH,
    ):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_asset_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max(1, num_hands),
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.num_hands = max(1, num_hands)

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style="white"):
        """Draws the custom skeletons manually using openCV matrices."""
        if keypoint_2d is None:
            return image

        h, w, _ = image.shape

        if style == "default":
            line_color = (0, 255, 0)
            dot_color = (0, 0, 255)
        else:
            line_color = (240, 240, 240)
            dot_color = (255, 48, 48)

        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (int(keypoint_2d[start_idx].x * w), int(keypoint_2d[start_idx].y * h))
            pt2 = (int(keypoint_2d[end_idx].x * w), int(keypoint_2d[end_idx].y * h))
            cv2.line(image, pt1, pt2, line_color, thickness=2)

        for lm in keypoint_2d:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), radius=4, color=dot_color, thickness=-1)

        return image

    def detect_all(
        self,
        rgb: np.ndarray,
        timestamp_ms: int,
        operator2mano_by_label: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, HandResult]:
        """Return detections keyed by MediaPipe handedness label (Left/Right)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

        if not results.hand_landmarks:
            return {}

        parsed: Dict[str, HandResult] = {}
        for i in range(len(results.hand_landmarks)):
            label = results.handedness[i][0].category_name
            if label in parsed:
                continue

            keypoint_3d = results.hand_world_landmarks[i]
            keypoint_2d = results.hand_landmarks[i]
            keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
            keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
            wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)

            operator2mano = np.eye(3)
            if operator2mano_by_label and label in operator2mano_by_label:
                operator2mano = operator2mano_by_label[label]

            joint_pos = keypoint_3d_array @ wrist_rot @ operator2mano
            parsed[label] = HandResult(
                joint_pos=joint_pos,
                keypoint_2d=keypoint_2d,
                wrist_rot=wrist_rot,
            )
        return parsed

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d[i].x
            keypoint[i][1] = keypoint_3d[i].y
            keypoint[i][2] = keypoint_3d[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d[i].x
            keypoint[i][1] = keypoint_2d[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        x_vector = points[0] - points[2]

        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)
        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        return np.stack([x, normal, z], axis=1)


class SingleHandDetector(HandDetector):
    """Backward-compatible single-hand detector with robot-side hand type selection."""

    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        super().__init__(
            num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    def detect(self, rgb, timestamp_ms: int):
        operator_map = {self.detected_hand_type: self.operator2mano}
        results = self.detect_all(rgb, timestamp_ms, operator_map)
        if self.detected_hand_type not in results:
            return 0, None, None, None
        hand = results[self.detected_hand_type]
        return 1, hand.joint_pos, hand.keypoint_2d, hand.wrist_rot
