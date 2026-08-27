"""Per-hand teleop state and SDK helpers for Sharpa Wave webcam control.

Pipeline per frame:
  align → calibration.apply → [right flip] → [landmark scale if YAML 1.0] → retarget

When YAML ``scaling_factor == 1.0``, frozen palm-span and per-finger MCP→tip
scales are computed once from calibration neutral vs URDF and applied before
retargeting (optimizer scaling stays 1.0).

When ``scaling_factor`` is any other value, landmark scaling is skipped and
the YAML factor is passed through to dex-retargeting unchanged.

Wireframe / snapshots use post-calibration landmarks (before landmark scaling).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_calibration import HandCalibration, align_landmarks, build_ref_value
from loguru import logger
from single_hand_detector import (
    OPERATOR2MANO_LEFT,
    OPERATOR2MANO_RIGHT,
    mediapipe_label_for_side,
)

if TYPE_CHECKING:
    from hand_visualization import HandVisualizer

HandSideName = Literal["left", "right"]

# ---------------------------------------------------------------------------
# Defaults — this module lives in supplements/; project root is one level up.
# ---------------------------------------------------------------------------
SUPPLEMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUPPLEMENTS_DIR.parent
DEFAULT_URDF_WAVE01 = REPO_ROOT / "sharpa-urdf-usd-xml" / "wave_01"
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "calibration"

_JOINT_SUFFIXES = [
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
]

# MediaPipe landmark indices.
_WRIST = 0
_INDEX_MCP = 5
_PINKY_MCP = 17

# Palm bases placed with the uniform span-derived scale (wrist → these).
_PALM_BASES = (1, 5, 9, 13, 17)  # thumb CMC + four finger MCPs

# name → (mp_mcp, mp_tip, robot_base_suffix, robot_tip_suffix)
# Finger scale = ||robot_base→tip|| / ||human MCP→tip||.
_FINGERS: Dict[str, Tuple[int, int, str, str]] = {
    "thumb": (1, 4, "thumb_MC", "thumb_fingertip"),
    "index": (5, 8, "index_PP", "index_fingertip"),
    "middle": (9, 12, "middle_PP", "middle_fingertip"),
    "ring": (13, 16, "ring_PP", "ring_fingertip"),
    "pinky": (17, 20, "pinky_PP", "pinky_fingertip"),
}

# Full MediaPipe chains from base through tip (for applying the finger scale).
_FINGER_CHAINS: Dict[str, Tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

_SCALE_CLAMP = (0.5, 2.5)
_EPS = 1e-9

_MP_LANDMARK_LABELS: Dict[int, str] = {
    0: "wrist",
    1: "thumb_CMC",
    2: "thumb_MCP",
    3: "thumb_IP",
    4: "thumb_tip",
    5: "index_MCP",
    6: "index_PIP",
    7: "index_DIP",
    8: "index_tip",
    9: "middle_MCP",
    10: "middle_PIP",
    11: "middle_DIP",
    12: "middle_tip",
    13: "ring_MCP",
    14: "ring_PIP",
    15: "ring_DIP",
    16: "ring_tip",
    17: "pinky_MCP",
    18: "pinky_PIP",
    19: "pinky_DIP",
    20: "pinky_tip",
}


def parse_side_from_path(path: str) -> HandSideName:
    lower = path.lower()
    if "right" in lower:
        return "right"
    if "left" in lower:
        return "left"
    raise ValueError(
        f"Cannot infer hand side from config path '{path}'. "
        "Filename must contain 'left' or 'right'."
    )


def parse_config_paths(config_paths: Tuple[str, ...]) -> Dict[HandSideName, str]:
    if not config_paths:
        raise ValueError("At least one --config-path is required.")
    if len(config_paths) > 2:
        raise ValueError("At most two config paths are supported (left and/or right).")

    parsed: Dict[HandSideName, str] = {}
    for path in config_paths:
        side = parse_side_from_path(path)
        if side in parsed:
            raise ValueError(f"Duplicate config for {side} hand: {path}")
        parsed[side] = path
    return parsed


def build_joint_names(side: HandSideName) -> List[str]:
    prefix = f"{side}_"
    return [prefix + suffix for suffix in _JOINT_SUFFIXES]


def mediapipe_label_for_side(side: HandSideName, *, selfie: bool = False) -> str:
    from single_hand_detector import mediapipe_label_for_side as _label

    return _label(side, selfie=selfie)


def operator2mano_for_side(side: HandSideName) -> np.ndarray:
    return OPERATOR2MANO_LEFT if side == "left" else OPERATOR2MANO_RIGHT


def resolve_robot_dir(side: HandSideName, base_robot_dir: str) -> str:
    """Return the directory that contains ``{side}_sharpa_wave.urdf``.

    ``base_robot_dir`` may be either the side folder
    (``.../wave_01/left_sharpa_wave``) or its parent (``.../wave_01``).
    """
    base = Path(base_robot_dir).expanduser().resolve()
    side_name = f"{side}_sharpa_wave"
    other_name = "right_sharpa_wave" if side == "left" else "left_sharpa_wave"
    default = DEFAULT_URDF_WAVE01 / side_name

    if base.name == side_name:
        return str(base)
    if base.name == other_name:
        return str(base.parent / side_name)

    candidate = base / side_name
    if candidate.is_dir():
        return str(candidate)

    base_str = str(base)
    if other_name in base_str:
        return base_str.replace(other_name, side_name)
    if side_name in base_str:
        return base_str
    return str(default)


def urdf_path_for_side(side: HandSideName, robot_dir: str) -> str:
    robot = Path(robot_dir)
    if robot.name != f"{side}_sharpa_wave":
        robot = Path(resolve_robot_dir(side, str(robot)))
    return str(robot / f"{side}_sharpa_wave.urdf")


def calibration_path_for_side(side: HandSideName) -> Path:
    DEFAULT_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CALIBRATION_DIR / f"hand_calibration_{side}.json"


def initialize_wave(wave) -> bool:
    from sharpa import ControlMode, ControlSource

    error = wave.set_control_mode(ControlMode.POSITION)
    if error.code != 0:
        logger.error(f"Failed to set control mode: {error.message}")
        return False
    error = wave.set_speed_coeff(0.3)
    if error.code != 0:
        logger.error(f"Failed to set speed coeff: {error.message}")
        return False
    error = wave.set_current_coeff(0.6)
    if error.code != 0:
        logger.error(f"Failed to set current coeff: {error.message}")
        return False
    error = wave.set_control_source(ControlSource.SDK)
    if error.code != 0:
        logger.error(f"Failed to set control source: {error.message}")
        return False
    return True


def _sdk_hand_side_name(info, wave) -> Optional[HandSideName]:
    try:
        from sharpa import HandSide
    except ImportError:
        return None

    if hasattr(info, "hand_side"):
        return "left" if info.hand_side == HandSide.LEFT else "right"
    if hasattr(wave, "get_hand_side"):
        hs = wave.get_hand_side()
        return "left" if hs == HandSide.LEFT else "right"
    return None


def connect_wave_hands(
    manager,
    expected_sides: List[HandSideName],
) -> Dict[HandSideName, Tuple[object, str]]:
    """Connect physical Sharpa hands for each expected side."""
    time.sleep(1.0)
    connected: Dict[HandSideName, Tuple[object, str]] = {}

    if hasattr(manager, "get_all_devices"):
        try:
            from sharpa import DeviceType

            device_infos = manager.get_all_devices()
            hand_infos = [
                info for info in device_infos
                if getattr(info, "device_type", None) == DeviceType.HAND
            ]
            for info in hand_infos:
                sn = info.sn
                wave = manager.connect(sn)
                side = _sdk_hand_side_name(info, wave)
                if side is None:
                    logger.warning(f"Device {sn}: could not determine hand side, skipping")
                    continue
                if side not in expected_sides:
                    logger.info(f"Device {sn} is {side} hand but not requested; skipping")
                    continue
                if side in connected:
                    logger.warning(f"Duplicate {side} device {sn}; keeping first")
                    continue
                connected[side] = (wave, sn)
                logger.info(f"Connected {side} hand: {sn}")
        except Exception as exc:
            logger.warning(f"get_all_devices failed ({exc}); falling back to serial list")

    if not connected:
        serials = manager.get_all_device_sn()
        if not serials:
            return connected
        if len(expected_sides) == 1 and len(serials) >= 1:
            side = expected_sides[0]
            sn = serials[0]
            connected[side] = (manager.connect(sn), sn)
            logger.info(f"Connected {side} hand (fallback): {sn}")
        elif len(expected_sides) == 2 and len(serials) >= 2:
            for side, sn in zip(expected_sides, serials[:2]):
                connected[side] = (manager.connect(sn), sn)
                logger.info(f"Connected {side} hand (fallback order): {sn}")

    return connected


@dataclass
class HandScaleFactors:
    """Frozen scales from calibration neutral vs URDF."""

    s_palm: float
    finger_scales: Dict[str, float]  # thumb/index/... → MCP→tip scale
    human_span: float
    robot_span: float
    human_finger_lengths: Dict[str, float]
    robot_finger_lengths: Dict[str, float]

    def summary_lines(self) -> List[str]:
        lines = [
            f"Palm span scale (uniform wrist→MCP): {self.s_palm:.4f} "
            f"(human knuckle span {self.human_span:.4f} m → "
            f"robot {self.robot_span:.4f} m)",
            "  Applies to MediaPipe bases: "
            + ", ".join(f"{_MP_LANDMARK_LABELS[i]}({i})" for i in _PALM_BASES),
            "Per-finger scales (MCP→fingertip, applied to whole finger chain):",
        ]
        for name in ("thumb", "index", "middle", "ring", "pinky"):
            mp_mcp, mp_tip, rob_base, rob_tip = _FINGERS[name]
            s = self.finger_scales[name]
            human_len = self.human_finger_lengths[name]
            robot_len = self.robot_finger_lengths[name]
            human = (
                f"{_MP_LANDMARK_LABELS[mp_mcp]}→{_MP_LANDMARK_LABELS[mp_tip]}"
            )
            lines.append(
                f"  {name:7s}  {human:22s}  "
                f"robot {rob_base}→{rob_tip:16s}  "
                f"human={human_len:.4f}m robot={robot_len:.4f}m  scale={s:.4f}"
            )
        return lines


def _clamp_scale(value: float) -> float:
    return float(np.clip(value, _SCALE_CLAMP[0], _SCALE_CLAMP[1]))


def _link_positions_at_q0(robot, link_names: List[str]) -> Dict[str, np.ndarray]:
    """FK at robot neutral; return world positions for named links."""
    q0 = np.asarray(robot.q0, dtype=np.float64).reshape(-1)
    robot.compute_forward_kinematics(q0)
    out: Dict[str, np.ndarray] = {}
    for name in link_names:
        idx = robot.get_link_index(name)
        pose = robot.get_link_pose(idx)
        out[name] = np.asarray(pose[:3, 3], dtype=np.float64).copy()
    return out


def compute_hand_scale_factors(
    side: HandSideName,
    corrected_neutral: np.ndarray,
    robot,
) -> HandScaleFactors:
    """Compute frozen palm-span + per-finger MCP→tip scales."""
    if corrected_neutral.shape != (21, 3):
        raise ValueError(
            f"Expected neutral landmarks (21, 3), got {corrected_neutral.shape}"
        )

    human_span = float(
        np.linalg.norm(
            corrected_neutral[_INDEX_MCP] - corrected_neutral[_PINKY_MCP]
        )
    )
    if human_span < _EPS:
        raise RuntimeError("Human knuckle span near zero; calibration landmarks invalid.")

    index_pp = f"{side}_index_PP"
    pinky_pp = f"{side}_pinky_PP"
    needed = {index_pp, pinky_pp}
    for _, _, rob_base, rob_tip in _FINGERS.values():
        needed.add(f"{side}_{rob_base}")
        needed.add(f"{side}_{rob_tip}")

    positions = _link_positions_at_q0(robot, sorted(needed))
    robot_span = float(np.linalg.norm(positions[index_pp] - positions[pinky_pp]))
    if robot_span < _EPS:
        raise RuntimeError("Robot knuckle span near zero; check URDF frames.")

    s_palm = _clamp_scale(robot_span / human_span)

    finger_scales: Dict[str, float] = {}
    human_finger_lengths: Dict[str, float] = {}
    robot_finger_lengths: Dict[str, float] = {}
    for name, (mp_mcp, mp_tip, rob_base, rob_tip) in _FINGERS.items():
        human_len = float(
            np.linalg.norm(corrected_neutral[mp_tip] - corrected_neutral[mp_mcp])
        )
        robot_len = float(
            np.linalg.norm(
                positions[f"{side}_{rob_tip}"] - positions[f"{side}_{rob_base}"]
            )
        )
        human_finger_lengths[name] = human_len
        robot_finger_lengths[name] = robot_len
        if human_len < _EPS:
            logger.warning(
                f"{side}: human {name} MCP→tip near zero; using scale 1.0"
            )
            finger_scales[name] = 1.0
        else:
            finger_scales[name] = _clamp_scale(robot_len / human_len)

    return HandScaleFactors(
        s_palm=s_palm,
        finger_scales=finger_scales,
        human_span=human_span,
        robot_span=robot_span,
        human_finger_lengths=human_finger_lengths,
        robot_finger_lengths=robot_finger_lengths,
    )


def scale_hand_chain(
    landmarks: np.ndarray,
    scales: HandScaleFactors,
) -> np.ndarray:
    """Rebuild the hand with palm scale then one MCP→tip scale per finger.

    Palm: wrist→MCP rays scaled by ``s_palm``.
    Fingers: all joints from MCP to tip scaled about the (already palm-scaled)
    MCP by that finger's single scale factor.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (21, 3):
        raise ValueError(f"Expected landmarks (21, 3), got {lm.shape}")

    out = lm.copy()
    wrist = lm[_WRIST]
    out[_WRIST] = wrist

    s_palm = scales.s_palm
    for mcp in _PALM_BASES:
        out[mcp] = wrist + s_palm * (lm[mcp] - wrist)

    for name, chain in _FINGER_CHAINS.items():
        mcp = chain[0]
        s = scales.finger_scales.get(name, 1.0)
        mcp_pos = out[mcp]
        for idx in chain[1:]:
            out[idx] = mcp_pos + s * (lm[idx] - lm[mcp])

    return out


@dataclass
class TeleopHand:
    """One hand: calibrate → [landmark-scale if YAML 1.0] → retarget → command."""

    side: HandSideName
    config_path: str
    robot_dir: str
    retargeting: object
    retargeting_type: str
    target_link_human_indices: np.ndarray
    retargeting_joint_names: List[str]
    retargeting_to_sharpa: np.ndarray
    calibration: HandCalibration
    mediapipe_label: str
    operator2mano: np.ndarray
    use_landmark_scaling: bool
    scales: Optional[HandScaleFactors] = None
    wave: Optional[object] = None
    device_sn: Optional[str] = None
    visualizer: Optional["HandVisualizer"] = field(default=None, repr=False)

    @property
    def urdf_full_path(self) -> str:
        return urdf_path_for_side(self.side, self.robot_dir)

    def process_frame(self, joint_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if joint_pos is None:
            return None
        ref_value = build_ref_value(
            joint_pos,
            self.retargeting_type,
            self.target_link_human_indices,
        )
        return self.retargeting.retarget(ref_value)

    def solve_from_joint_pos(
        self, joint_pos: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Align, calibrate, scale landmarks, retarget.

        Returns post-calibration landmarks for visualization and ``qpos`` from
        the length-scaled chain. Safe to run in a worker thread (no SDK I/O).
        """
        corrected_array = self.calibration.apply(align_landmarks(joint_pos))
        if self.side == "right":
            corrected_array[:, 0] *= -1
            corrected_array[:, 1] *= -1

        if self.use_landmark_scaling:
            if self.scales is None:
                raise RuntimeError(
                    f"{self.side}: landmark scaling enabled but scales missing"
                )
            retarget_landmarks = scale_hand_chain(corrected_array, self.scales)
        else:
            retarget_landmarks = corrected_array
        qpos = self.process_frame(retarget_landmarks)
        return corrected_array, qpos

    def command_robot(
        self,
        qpos: Optional[np.ndarray],
        *,
        enable_interp: bool,
        control_enabled: bool,
    ) -> None:
        if self.wave is None:
            return
        if not control_enabled or qpos is None:
            self.wave.set_joint_position([0.0] * 22, enable_interp)
            return
        positions = [float(x) for x in qpos[self.retargeting_to_sharpa]]
        error = self.wave.set_joint_position(positions, enable_interp)
        if error.code != 0:
            logger.warning(
                f"{self.side} hand failed to set joint position: {error.message}"
            )


def build_teleop_hand(
    side: HandSideName,
    config_path: str,
    base_robot_dir: str,
    *,
    wave: Optional[object] = None,
    device_sn: Optional[str] = None,
    calibration: HandCalibration,
) -> TeleopHand:
    robot_dir = resolve_robot_dir(side, base_robot_dir)
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Loading {side} retargeting from {config_path} (urdf dir {robot_dir})")
    config = RetargetingConfig.load_from_file(config_path)
    retargeting = config.build()
    use_landmark_scaling = config.scaling_factor == 1.0

    retargeting_type = retargeting.optimizer.retargeting_type
    target_link_human_indices = retargeting.optimizer.target_link_human_indices
    joint_names = build_joint_names(side)
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sharpa = np.array(
        [retargeting_joint_names.index(name) for name in joint_names],
        dtype=int,
    )

    scales: Optional[HandScaleFactors] = None
    if use_landmark_scaling:
        # Lengths from the same space used at runtime (post-apply, pre right-flip).
        corrected_neutral = calibration.apply(calibration.neutral_landmarks.copy())
        scales = compute_hand_scale_factors(
            side,
            corrected_neutral,
            retargeting.optimizer.robot,
        )
        for line in scales.summary_lines():
            logger.info(f"{side} {line}")
    else:
        logger.info(
            f"{side} using YAML scaling_factor={config.scaling_factor} "
            "(landmark scaling disabled)"
        )

    return TeleopHand(
        side=side,
        config_path=config_path,
        robot_dir=robot_dir,
        retargeting=retargeting,
        retargeting_type=retargeting_type,
        target_link_human_indices=target_link_human_indices,
        retargeting_joint_names=retargeting_joint_names,
        retargeting_to_sharpa=retargeting_to_sharpa,
        calibration=calibration,
        mediapipe_label=mediapipe_label_for_side(side),
        operator2mano=operator2mano_for_side(side),
        use_landmark_scaling=use_landmark_scaling,
        scales=scales,
        wave=wave,
        device_sn=device_sn,
    )
