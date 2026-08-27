"""Flat-hand MediaPipe calibration for wrist-aligned landmark correction."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from frame_queue_utils import get_latest_frame
from loguru import logger

FINGERS = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}

# Thumb uses a different axis; MCP/PIP calibration applies to the four fingers only.
CALIBRATION_FINGERS = {
    name: indices for name, indices in FINGERS.items() if name != "Thumb"
}


# ---------------------------------------------------------------------------
# Defaults — edit these if your machine layout differs from this repo.
# This module lives in supplements/; project root is one level up.
# ---------------------------------------------------------------------------
SUPPLEMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUPPLEMENTS_DIR.parent
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "calibration"


def calibration_path_for_side(side: str) -> Path:
    DEFAULT_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CALIBRATION_DIR / f"hand_calibration_{side}.json"


DEFAULT_CALIBRATION_PATH = calibration_path_for_side("left")

REST_REF_VALUE_NORM_THRESHOLD = 0.005


def align_landmarks(joint_pos: np.ndarray) -> np.ndarray:
    """Wrist-center and rotate landmarks into the palm-aligned frame."""
    wrist = joint_pos[0]
    index = joint_pos[5]
    middle = joint_pos[9]
    pinky = joint_pos[17]

    x = middle - wrist
    x /= np.linalg.norm(x)

    y_temp = pinky - index
    y_temp /= np.linalg.norm(y_temp)

    z = np.cross(x, y_temp)
    z /= np.linalg.norm(z)

    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    #maps rotation matrix to the coordinate system expected by the Sharpa Wave hand
    r_hand = np.column_stack((z, y, x))
    #centers joints around the wrist so they are uniformly comparable
    centered_joints = joint_pos - joint_pos[0]
    return centered_joints @ r_hand


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def compute_bend_angles(joint_pos: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute PIP and DIP bend angles per finger from raw MediaPipe landmarks."""
    bends: Dict[str, Dict[str, float]] = {}
    for name, (mcp, pip, dip, tip) in CALIBRATION_FINGERS.items():
        bends[name] = {
            "mcp": angle(joint_pos[0], joint_pos[mcp], joint_pos[pip]),
            "pip": angle(joint_pos[mcp], joint_pos[pip], joint_pos[dip]),
            "dip": angle(joint_pos[pip], joint_pos[dip], joint_pos[tip])
        }
    return bends


def _project_onto_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> np.ndarray:
    direction = line_end - line_start
    length_sq = float(np.dot(direction, direction))
    if length_sq < 1e-12:
        return line_start.copy()
    t = float(np.dot(point - line_start, direction) / length_sq)
    return line_start + t * direction


def _palm_normal(landmarks: np.ndarray) -> np.ndarray:
    """Unit palm normal from wrist and MCP triangle (aligned-frame landmarks)."""
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    normal = np.cross(index_mcp - wrist, middle_mcp - wrist)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return normal / norm


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-9 or target_norm < 1e-9:
        return np.eye(3, dtype=np.float64)

    a = source / source_norm
    b = target / target_norm
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        return _axis_angle_matrix(axis, np.pi)

    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return _axis_angle_matrix(axis, angle)


def _rotation_zero_normal_component(vector: np.ndarray, palm_normal: np.ndarray) -> np.ndarray:
    """Rotation that removes the palm-normal component from vector direction."""
    n = palm_normal / np.linalg.norm(palm_normal)
    flattened = vector - np.dot(vector, n) * n
    return _rotation_from_to(vector, flattened)


def compute_mcp_pip_vectors(landmarks: np.ndarray) -> Dict[str, np.ndarray]:
    """MCP->PIP vector for each calibrated finger in aligned landmark coordinates."""
    vectors: Dict[str, np.ndarray] = {}
    for name, (mcp, pip, _, _) in CALIBRATION_FINGERS.items():
        vectors[name] = landmarks[pip] - landmarks[mcp]
    return vectors


def apply_mcp_rotation_correction(
    landmarks: np.ndarray,
    mcp_rotations: Dict[str, np.ndarray],
) -> np.ndarray:
    """Rotate each calibrated finger chain about its MCP to flatten MCP->PIP."""
    corrected = landmarks.copy()
    for name, (mcp, pip, dip, tip) in CALIBRATION_FINGERS.items():
        rotation = mcp_rotations.get(name)
        if rotation is None:
            continue
        pivot = landmarks[mcp]
        for idx in (pip, dip, tip):
            corrected[idx] = pivot + rotation @ (landmarks[idx] - pivot)
    return corrected


def straighten_fingers(landmarks: np.ndarray) -> np.ndarray:
    """Straighten each finger chain by projecting joints onto MCP-to-TIP line."""
    straight = landmarks.copy()
    straight[0] = landmarks[0]

    for mcp, pip, dip, tip in CALIBRATION_FINGERS.values():
        anchor = landmarks[mcp]
        tip_point = landmarks[tip]
        straight[mcp] = anchor
        straight[pip] = _project_onto_line(landmarks[pip], anchor, tip_point)
        straight[dip] = _project_onto_line(landmarks[dip], anchor, tip_point)
        straight[tip] = tip_point

    return straight


def build_ref_value(
    rotated_array: np.ndarray,
    retargeting_type: str,
    target_link_human_indices: np.ndarray,
) -> np.ndarray:
    if retargeting_type == "POSITION":
        return rotated_array[target_link_human_indices, :]
    origin_indices = target_link_human_indices[0, :]
    task_indices = target_link_human_indices[1, :]
    return rotated_array[task_indices, :] - rotated_array[origin_indices, :]



def _array_to_list(array: np.ndarray) -> List[List[float]]:
    return array.astype(float).tolist()


def _list_to_array(values: List[List[float]]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


@dataclass
class HandCalibration:
    neutral_landmarks: np.ndarray
    template_flat_landmarks: np.ndarray
    spurious_offset: np.ndarray
    neutral_ref_value: np.ndarray
    bend_angles_at_rest: Dict[str, Dict[str, float]]
    mcp_pip_vectors: Dict[str, np.ndarray]
    mcp_rotations: Dict[str, np.ndarray]
    palm_normal_at_rest: np.ndarray
    timestamp: float
    num_frames: int
    retargeting_type: str = "VECTOR"
    target_link_human_indices: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_neutral(
        cls,
        neutral_landmarks: np.ndarray,
        num_frames: int,
        retargeting_type: str,
        target_link_human_indices: np.ndarray,
    ) -> "HandCalibration":
        palm_normal = _palm_normal(neutral_landmarks)
        mcp_pip_vectors = compute_mcp_pip_vectors(neutral_landmarks)
        mcp_rotations = {
            name: _rotation_zero_normal_component(vector, palm_normal)
            for name, vector in mcp_pip_vectors.items()
        }

        mcp_corrected = apply_mcp_rotation_correction(neutral_landmarks, mcp_rotations)
        template = straighten_fingers(mcp_corrected)
        spurious = mcp_corrected - template
        neutral_ref = build_ref_value(
            mcp_corrected, retargeting_type, target_link_human_indices
        )
        return cls(
            neutral_landmarks=neutral_landmarks.copy(),
            template_flat_landmarks=template,
            spurious_offset=spurious,
            neutral_ref_value=neutral_ref,
            bend_angles_at_rest=compute_bend_angles(neutral_landmarks),
            mcp_pip_vectors=mcp_pip_vectors,
            mcp_rotations=mcp_rotations,
            palm_normal_at_rest=palm_normal,
            timestamp=time.time(),
            num_frames=num_frames,
            retargeting_type=retargeting_type,
            target_link_human_indices=target_link_human_indices.copy(),
        )

    def apply(self, rotated_array: np.ndarray) -> np.ndarray:
        mcp_corrected = apply_mcp_rotation_correction(rotated_array, self.mcp_rotations)
        return mcp_corrected - self.spurious_offset

    def summary_lines(self) -> List[str]:
        lines = [f"Calibration from {self.num_frames} frames (4 fingers, thumb excluded):"]
        for finger in CALIBRATION_FINGERS:
            angles = self.bend_angles_at_rest[finger]
            vector = self.mcp_pip_vectors[finger]
            normal_component = float(np.dot(vector, self.palm_normal_at_rest))
            lines.append(
                f"  {finger:6s} MCP->PIP normal: {normal_component:+.4f} | "
                f"PIP rest: {angles['pip']:6.2f} deg | "
                f"DIP rest: {angles['dip']:6.2f} deg"
            )
        return lines

    def save(self, path: Path | str = DEFAULT_CALIBRATION_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "num_frames": self.num_frames,
            "retargeting_type": self.retargeting_type,
            "target_link_human_indices": (
                self.target_link_human_indices.tolist()
                if self.target_link_human_indices is not None
                else None
            ),
            "neutral_landmarks": _array_to_list(self.neutral_landmarks),
            "template_flat_landmarks": _array_to_list(self.template_flat_landmarks),
            "spurious_offset": _array_to_list(self.spurious_offset),
            "neutral_ref_value": _array_to_list(self.neutral_ref_value),
            "bend_angles_at_rest": self.bend_angles_at_rest,
            "mcp_pip_vectors": {
                name: vector.astype(float).tolist()
                for name, vector in self.mcp_pip_vectors.items()
            },
            "mcp_rotations": {
                name: matrix.astype(float).tolist()
                for name, matrix in self.mcp_rotations.items()
            },
            "palm_normal_at_rest": self.palm_normal_at_rest.astype(float).tolist(),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(f"Saved hand calibration to {path}")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CALIBRATION_PATH) -> "HandCalibration":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        indices = payload.get("target_link_human_indices")
        neutral_landmarks = _list_to_array(payload["neutral_landmarks"])
        palm_normal = payload.get("palm_normal_at_rest")
        mcp_pip_vectors_raw = payload.get("mcp_pip_vectors")
        mcp_rotations_raw = payload.get("mcp_rotations")

        if mcp_pip_vectors_raw is None or mcp_rotations_raw is None:
            palm_normal_arr = (
                np.asarray(palm_normal, dtype=np.float64)
                if palm_normal is not None
                else _palm_normal(neutral_landmarks)
            )
            mcp_pip_vectors = compute_mcp_pip_vectors(neutral_landmarks)
            mcp_rotations = {
                name: _rotation_zero_normal_component(vector, palm_normal_arr)
                for name, vector in mcp_pip_vectors.items()
            }
        else:
            mcp_pip_vectors = {
                name: np.asarray(vector, dtype=np.float64)
                for name, vector in mcp_pip_vectors_raw.items()
            }
            mcp_rotations = {
                name: np.asarray(matrix, dtype=np.float64)
                for name, matrix in mcp_rotations_raw.items()
            }
            palm_normal_arr = (
                np.asarray(palm_normal, dtype=np.float64)
                if palm_normal is not None
                else _palm_normal(neutral_landmarks)
            )

        return cls(
            neutral_landmarks=neutral_landmarks,
            template_flat_landmarks=_list_to_array(payload["template_flat_landmarks"]),
            spurious_offset=_list_to_array(payload["spurious_offset"]),
            neutral_ref_value=_list_to_array(payload["neutral_ref_value"]),
            bend_angles_at_rest=payload["bend_angles_at_rest"],
            mcp_pip_vectors=mcp_pip_vectors,
            mcp_rotations=mcp_rotations,
            palm_normal_at_rest=palm_normal_arr,
            timestamp=float(payload["timestamp"]),
            num_frames=int(payload["num_frames"]),
            retargeting_type=str(payload.get("retargeting_type", "VECTOR")),
            target_link_human_indices=(
                np.asarray(indices, dtype=int) if indices is not None else None
            ),
        )


def _landmark_frame_variance(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    stacked = np.stack(frames, axis=0)
    return float(np.mean(np.std(stacked, axis=0)))


def _draw_calibration_overlay(
    image: np.ndarray,
    *,
    side: str = "left",
    elapsed: float,
    duration_s: float,
    frame_count: int,
    min_frames: int,
    bend_angles: Optional[Dict[str, Dict[str, float]]],
    status: str,
) -> np.ndarray:
    overlay = image.copy()
    h, w = overlay.shape[:2]

    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, overlay)

    cv2.putText(
        overlay,
        f"Calibrate {side.upper()} hand - hold flat to camera (fingers extended)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        status,
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    progress = min(elapsed / duration_s, 1.0)
    bar_x, bar_y, bar_w, bar_h = 20, 72, w - 40, 18
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(
        overlay,
        (bar_x, bar_y),
        (bar_x + int(bar_w * progress), bar_y + bar_h),
        (0, 200, 0),
        -1,
    )
    cv2.putText(
        overlay,
        f"Collecting: {elapsed:.1f} / {duration_s:.1f}s ({frame_count} frames, need {min_frames})",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )

    if bend_angles is not None:
        y = 150
        for finger, angles in bend_angles.items():
            text = (
                f"{finger}: MCP PIP/DIP {angles['pip']:5.1f}/{angles['dip']:5.1f} deg"
            )
            cv2.putText(
                overlay,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 220, 120),
                1,
                cv2.LINE_AA,
            )
            y += 22

    return overlay


def run_calibration(
    detector,
    queue,
    *,
    side: str = "left",
    mediapipe_label: Optional[str] = None,
    retargeting_type: str,
    target_link_human_indices: np.ndarray,
    duration_s: float = 3.0,
    min_frames: int = 45,
    max_variance: float = 0.004,
    save_path: Optional[Path | str] = None,
    window_name: str = "hand_calibration",
) -> HandCalibration:
    """Capture a flat-hand baseline and build landmark correction offsets."""
    if save_path is None:
        save_path = calibration_path_for_side(side)
    if mediapipe_label is None:
        from single_hand_detector import mediapipe_label_for_side

        mediapipe_label = mediapipe_label_for_side(side)

    collected: List[np.ndarray] = []
    recent_for_stability: List[np.ndarray] = []
    start_time = time.time()
    last_bend_angles: Optional[Dict[str, Dict[str, float]]] = None

    logger.info(
        f"Starting {side} flat-hand calibration for {duration_s:.1f}s "
        f"(minimum {min_frames} stable frames)"
    )

    def _detect_target_hand(rgb, timestamp_ms):
        if hasattr(detector, "detect_all"):
            results = detector.detect_all(rgb, timestamp_ms)
            hand = results.get(mediapipe_label)
            if hand is None:
                return None, None
            return hand.joint_pos, hand.keypoint_2d
        _, joint_pos, keypoint_2d, _ = detector.detect(rgb, timestamp_ms)
        return joint_pos, keypoint_2d

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_s and len(collected) >= min_frames:
            break

        try:
            _, bgr, _ = get_latest_frame(queue, timeout=5.0)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty as exc:
            raise RuntimeError(
                "Camera timeout during calibration. Check your webcam device."
            ) from exc

        timestamp_ms = int(time.time() * 1000)
        joint_pos, keypoint_2d = _detect_target_hand(rgb, timestamp_ms)

        
        if keypoint_2d is not None:
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")

        if joint_pos is None:
            status = f"Waiting for {side.upper()} hand detection..."
            overlay = _draw_calibration_overlay(
                bgr,
                side=side,
                elapsed=elapsed,
                duration_s=duration_s,
                frame_count=len(collected),
                min_frames=min_frames,
                bend_angles=last_bend_angles,
                status=status,
            )
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(window_name)
                raise RuntimeError("Calibration aborted by user.")
            continue

        aligned = align_landmarks(joint_pos)
        recent_for_stability.append(aligned)
        if len(recent_for_stability) > 10:
            recent_for_stability.pop(0)

        variance = _landmark_frame_variance(recent_for_stability)
        last_bend_angles = compute_bend_angles(joint_pos)

        if variance <= max_variance:
            collected.append(aligned)
            status = "Collecting stable frames..."
        else:
            status = "Hold still - hand movement detected"

        overlay = _draw_calibration_overlay(
            bgr,
            side=side,
            elapsed=elapsed,
            duration_s=duration_s,
            frame_count=len(collected),
            min_frames=min_frames,
            bend_angles=last_bend_angles,
            status=status,
        )
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Calibration aborted by user.")

    if len(collected) < min_frames:
        cv2.destroyWindow(window_name)
        raise RuntimeError(
            f"{side.upper()} calibration failed: only {len(collected)} stable frames "
            f"collected, need at least {min_frames}. Hold your hand flat and still."
        )

    stacked = np.stack(collected, axis=0)
    neutral_landmarks = np.median(stacked, axis=0)

    calibration = HandCalibration.from_neutral(
        neutral_landmarks,
        num_frames=len(collected),
        retargeting_type=retargeting_type,
        target_link_human_indices=target_link_human_indices,
    )

    for line in calibration.summary_lines():
        logger.info(line)

    calibration.save(save_path)

    complete_overlay = _draw_calibration_overlay(
        bgr,
        side=side,
        elapsed=duration_s,
        duration_s=duration_s,
        frame_count=len(collected),
        min_frames=min_frames,
        bend_angles=calibration.bend_angles_at_rest,
        status=f"{side.upper()} calibration complete",
    )
    cv2.imshow(window_name, complete_overlay)
    cv2.waitKey(1000)
    cv2.destroyWindow(window_name)

    return calibration
