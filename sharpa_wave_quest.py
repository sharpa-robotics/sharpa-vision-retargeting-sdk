"""Sharpa Wave teleop driven by Meta Quest Hand Tracking Streamer (HTS) over UDP.

Point the HTS app at this PC's Wi-Fi IPv4 and
UDP port ``DEFAULT_HTS_PORT`` (default 9000).

Placeholder frames are only used for calibration / mode-select UI.
Teleop control is event-driven on new HTS packets (no 30 Hz blank-frame wait).
Quest UI during teleop: wireframe windows (mode 1/2) or a minimal quit window (mode 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults — edit these if your machine layout differs from this repo.
# Relative paths are anchored at this repository root.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
SUPPLEMENTS_DIR = REPO_ROOT / "supplements"
DEFAULT_SHARPA_SDK_PYTHON = "/opt/sharpa-wave-sdk/python"
DEFAULT_CONFIG_PATH = str(SUPPLEMENTS_DIR / "sharpa_wave_left.yml")
DEFAULT_ROBOT_DIR_PATH = str(REPO_ROOT / "sharpa-urdf-usd-xml" / "wave_01")
DEFAULT_HTS_HOST = "0.0.0.0"
DEFAULT_HTS_PORT = 9000
PLACEHOLDER_FRAME_SIZE = (480, 640, 3)  # H, W, C — blank canvas for calib / mode UI
QUEST_QUIT_WINDOW = "Quest teleop — press q to quit"
HTS_POSE_WAIT_TIMEOUT_S = 0.05

if str(SUPPLEMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPLEMENTS_DIR))
if DEFAULT_SHARPA_SDK_PYTHON not in sys.path:
    sys.path.insert(0, DEFAULT_SHARPA_SDK_PYTHON)

import multiprocessing
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from queue import Empty, Full
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tyro
from dex_retargeting.retargeting_config import RetargetingConfig
from frame_queue_utils import flush_queue, get_latest_frame
from hand_calibration import run_calibration
from hand_tracking_sdk import (
    ErrorPolicy,
    HandFilter,
    HandFrame,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
    convert_hand_frame_unity_left_to_right,
)
from hand_visualization import (
    HandSideSnapshot,
    HandVisualizer,
    SharedTeleopState,
    TeleopMode,
    WireframeRenderProcess,
    close_all_visualizers,
    prompt_teleop_mode,
    teleop_mode_label,
)
from loguru import logger
from sharpa import SharpaWaveManager
from single_hand_detector import (
    OPERATOR2MANO_LEFT,
    OPERATOR2MANO_RIGHT,
    HandDetector,
    HandResult,
)
from teleop_hand import (
    TeleopHand,
    build_teleop_hand,
    connect_wave_hands,
    initialize_wave,
    operator2mano_for_side,
    parse_config_paths,
    resolve_robot_dir,
)

print("Sharpa Wave Quest Teleop - Starting")


def quest_label_for_side(side: str) -> str:
    """HTS uses absolute Left/Right (no webcam selfie flip)."""
    return "Left" if side == "left" else "Right"


@dataclass
class _LatestHand:
    joint_pos: np.ndarray
    recv_ts: float


class QuestHandReceiver:
    """Background UDP listener keeping the latest left/right HTS landmarks."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HTS_HOST,
        port: int = DEFAULT_HTS_PORT,
        convert_unity_to_right_handed: bool = True,
    ) -> None:
        self._host = host
        self._port = port
        self._convert = convert_unity_to_right_handed
        self._lock = threading.Lock()
        self._latest: Dict[str, _LatestHand] = {}
        self._stop = threading.Event()
        self._new_pose_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._new_pose_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="quest-hts-udp",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Quest HTS UDP listener on {self._host}:{self._port} "
            "(set HTS app IP to this PC's Wi-Fi address)"
        )

    def stop(self) -> None:
        self._stop.set()
        self._new_pose_event.set()  # unblock waiters
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def wait_new_pose(self, timeout: float = HTS_POSE_WAIT_TIMEOUT_S) -> bool:
        """Block until a new HTS hand frame arrives (or timeout)."""
        return self._new_pose_event.wait(timeout=timeout)

    def clear_new_pose(self) -> None:
        self._new_pose_event.clear()

    def get_all_joint_pos(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {k: v.joint_pos.copy() for k, v in self._latest.items()}

    def _run(self) -> None:
        client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode.UDP,
                host=self._host,
                port=self._port,
                timeout_s=0.5,
                output=StreamOutput.FRAMES,
                hand_filter=HandFilter.BOTH,
                error_policy=ErrorPolicy.TOLERANT,
            )
        )
        try:
            for event in client.iter_events():
                if self._stop.is_set():
                    break
                if not isinstance(event, HandFrame):
                    continue
                frame = (
                    convert_hand_frame_unity_left_to_right(event)
                    if self._convert
                    else event
                )
                points = np.asarray(frame.landmarks.points, dtype=np.float64)
                if points.shape != (21, 3):
                    continue
                label = frame.side.value  # "Left" / "Right"

                # DEBUG: dump every packet

                now = time.time()
                if now - getattr(self, "_last_dbg", 0) > 0.2:  # 5 Hz
                    self._last_dbg = now
                    wrist = points[0]
                    span = float(np.linalg.norm(points[5] - points[17]))  # index–pinky MCP

                    
                    print(
                        f"[HTS] {label} seq={getattr(frame, 'sequence_id', '?')} "
                        f"wrist=({wrist[0]:+.4f},{wrist[1]:+.4f},{wrist[2]:+.4f}) "
                        f"palm_span={span:.4f} "
                        f"std={float(np.std(points)):.5f} "
                        f"min={points.min(axis=0)} max={points.max(axis=0)}"
                    )
                    # optional full dump (noisy):
                    print(points)
                with self._lock:
                    self._latest[label] = _LatestHand(
                        joint_pos=points,
                        recv_ts=time.time(),
                    )
                self._new_pose_event.set()
        except Exception:
            if not self._stop.is_set():
                logger.exception("Quest HTS receiver stopped with error")
        finally:
            logger.info("Quest HTS UDP listener stopped")


class QuestHandDetector:
    """MediaPipe-compatible detector that reads Quest HTS landmarks.

    ``rgb`` is ignored for pose. Used so ``run_calibration`` can stay unchanged.
    """

    draw_skeleton_on_image = staticmethod(HandDetector.draw_skeleton_on_image)

    def __init__(self, receiver: QuestHandReceiver) -> None:
        self._receiver = receiver

    def detect_all(
        self,
        rgb: np.ndarray,
        timestamp_ms: int,
        operator2mano_by_label: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, HandResult]:
        del rgb, timestamp_ms
        parsed: Dict[str, HandResult] = {}
        for label, points in self._receiver.get_all_joint_pos().items():
            joint_pos = points.copy()
            joint_pos = joint_pos - joint_pos[0:1, :]
            wrist_rot = HandDetector.estimate_frame_from_hand_points(joint_pos)
            if operator2mano_by_label and label in operator2mano_by_label:
                operator2mano = operator2mano_by_label[label]
            else:
                operator2mano = (
                    OPERATOR2MANO_LEFT if label == "Left" else OPERATOR2MANO_RIGHT
                )
            joint_pos = joint_pos @ wrist_rot @ operator2mano
            parsed[label] = HandResult(
                joint_pos=joint_pos,
                keypoint_2d=None,
                wrist_rot=wrist_rot,
            )
        return parsed


def _placeholder_bgr() -> np.ndarray:
    return np.zeros(PLACEHOLDER_FRAME_SIZE, dtype=np.uint8)


def _fetch_bgr_frame(
    queue: multiprocessing.Queue,
) -> Tuple[Optional[float], np.ndarray, int]:
    try:
        return get_latest_frame(queue, timeout=5.0)
    except Empty as exc:
        raise RuntimeError(
            "No placeholder frames from Quest UI producer. Is the process running?"
        ) from exc


def produce_placeholder_frames(
    queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    hz: float = 30.0,
) -> None:
    """Feed blank BGR frames so calibration / mode-select OpenCV loops keep ticking."""
    period = 1.0 / max(hz, 1.0)
    frame = _placeholder_bgr()
    while not stop_event.is_set():
        item = (time.time(), frame.copy())
        try:
            queue.put_nowait(item)
        except Full:
            try:
                queue.get_nowait()
                queue.put_nowait(item)
            except (Empty, Full):
                pass
        time.sleep(period)


def run_quest_visualization_loop(
    visualizers: Dict[str, HandVisualizer],
    shared_state: SharedTeleopState,
    stop_event: threading.Event,
    teleop_mode: TeleopMode,
) -> None:
    """Quest-only UI: wireframes in mode 1/2; minimal quit window in mode 3.

    Does not show the blank ``realtime_retargeting_demo`` webcam window.
    """
    show_wireframes = teleop_mode in (
        TeleopMode.BOTH,
        TeleopMode.VISUALIZATION_ONLY,
    )

    if not show_wireframes:
        canvas = np.zeros((80, 420, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "Quest teleop (mode 3) — press q to quit",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.namedWindow(QUEST_QUIT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(QUEST_QUIT_WINDOW, 420, 80)
        try:
            while not stop_event.is_set():
                cv2.imshow(QUEST_QUIT_WINDOW, canvas)
                if (cv2.waitKey(30) & 0xFF) == ord("q"):
                    stop_event.set()
                    break
        finally:
            try:
                cv2.destroyWindow(QUEST_QUIT_WINDOW)
            except cv2.error:
                pass
        return

    last_wireframe_frame = -1
    render_process = next(
        (v.render_process for v in visualizers.values() if v.render_process is not None),
        None,
    )
    wireframe_interval = (
        render_process.wireframe_interval if render_process is not None else 5
    )

    while not stop_event.is_set():
        snapshot = shared_state.get_latest()
        if snapshot is not None and render_process is not None:
            if (
                snapshot.frame_counter % wireframe_interval == 0
                and snapshot.frame_counter != last_wireframe_frame
            ):
                batch: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for side, visualizer in visualizers.items():
                    if not visualizer.show_wireframes:
                        continue
                    side_snap = snapshot.hands.get(side)
                    if (
                        side_snap is not None
                        and side_snap.hand_detected
                        and side_snap.corrected_array is not None
                        and side_snap.qpos is not None
                    ):
                        batch[side] = (side_snap.corrected_array, side_snap.qpos)
                if batch:
                    render_process.submit(batch, snapshot.frame_counter)
                    last_wireframe_frame = snapshot.frame_counter

            for side, visualizer in visualizers.items():
                wireframe_bgr = visualizer.poll_wireframe_image()
                if wireframe_bgr is not None:
                    cv2.imshow(visualizer.wireframe_window, wireframe_bgr)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            stop_event.set()
            break
        time.sleep(0.001)


def start_retargeting(
    queue: multiprocessing.Queue,
    robot_dir: str,
    config_paths: Tuple[str, ...],
    stop_event: multiprocessing.Event,
    hts_host: str,
    hts_port: int,
):
    visualizers: Dict[str, HandVisualizer] = {}
    teleop_stop = threading.Event()
    quest_receiver = QuestHandReceiver(host=hts_host, port=hts_port)
    render_process = None
    try:
        configs = parse_config_paths(config_paths)
        expected_sides = list(configs.keys())
        logger.info(f"Requested teleop sides (Quest HTS): {expected_sides}")

        quest_receiver.start()
        detector = QuestHandDetector(quest_receiver)
        # HTS labels are absolute Left/Right (not MediaPipe selfie-flipped).
        operator_map = {
            quest_label_for_side(side): operator2mano_for_side(side)
            for side in expected_sides
        }

        manager = SharpaWaveManager.get_instance()
        connected = connect_wave_hands(manager, expected_sides)

        if len(expected_sides) == 1 and expected_sides[0] not in connected:
            raise RuntimeError(
                f"No Sharpa device found for {expected_sides[0]} hand teleop."
            )

        active_hands: Dict[str, TeleopHand] = {}
        for side, config_path in configs.items():
            wave, device_sn = connected.get(side, (None, None))
            if wave is None:
                logger.warning(
                    f"No Sharpa device for {side} hand; continuing without robot control."
                )
            else:
                if not initialize_wave(wave):
                    raise RuntimeError(f"Failed to initialize {side} hand")
                wave.start()

            side_robot_dir = resolve_robot_dir(side, robot_dir)
            RetargetingConfig.set_default_urdf_dir(str(side_robot_dir))
            retargeting = RetargetingConfig.load_from_file(config_path).build()

            flush_queue(queue)
            logger.info(
                f"Hold {side.upper()} hand flat in Quest view for calibration "
                f"(listening UDP {hts_host}:{hts_port})"
            )
            calibration = run_calibration(
                detector,
                queue,
                side=side,
                mediapipe_label=quest_label_for_side(side),
                retargeting_type=retargeting.optimizer.retargeting_type,
                target_link_human_indices=retargeting.optimizer.target_link_human_indices,
            )
            hand = build_teleop_hand(
                side,
                config_path,
                robot_dir,
                wave=wave,
                device_sn=device_sn,
                calibration=calibration,
            )
            # Override webcam-flipped label with HTS absolute side label.
            hand.mediapipe_label = quest_label_for_side(side)
            active_hands[side] = hand
            logger.info(
                f"Active {side} hand (Quest label={hand.mediapipe_label})"
                + (f" (SN {device_sn})" if device_sn else " (no device)")
            )

        if not active_hands:
            raise RuntimeError("No teleop hands could be initialized.")

        flush_queue(queue)
        _, mode_frame, _ = _fetch_bgr_frame(queue)
        teleop_mode = prompt_teleop_mode(mode_frame)
        logger.info(f"Teleop mode selected: {teleop_mode_label(teleop_mode)}")

        control_hand = teleop_mode != TeleopMode.VISUALIZATION_ONLY
        if teleop_mode in (TeleopMode.BOTH, TeleopMode.VISUALIZATION_ONLY):
            side_configs = [
                (side, hand.urdf_full_path) for side, hand in active_hands.items()
            ]
            render_process = WireframeRenderProcess(side_configs)
            render_process.start()
            for side, hand in active_hands.items():
                hand.visualizer = HandVisualizer(
                    teleop_mode,
                    side=side,
                    urdf_path=hand.urdf_full_path,
                    render_process=render_process,
                )
                visualizers[side] = hand.visualizer

        shared_state = SharedTeleopState()
        frame_counter = 0
        no_hand_warn_interval = 2.0
        last_no_hand_warn: Dict[str, float] = {side: 0.0 for side in active_hands}
        enable_interpolation_mode = True
        retarget_executor = ThreadPoolExecutor(
            max_workers=max(1, len(active_hands)),
            thread_name_prefix="retarget",
        )
        # Published only to satisfy SharedTeleopState; Quest UI never shows it.
        publish_bgr = _placeholder_bgr()

        def control_loop() -> None:
            nonlocal frame_counter
            while not teleop_stop.is_set():
                quest_receiver.wait_new_pose(HTS_POSE_WAIT_TIMEOUT_S)
                quest_receiver.clear_new_pose()
                if teleop_stop.is_set():
                    break

                frame_counter += 1
                timestamp_ms = int(time.time() * 1000)
                detections = detector.detect_all(
                    publish_bgr,
                    timestamp_ms,
                    operator2mano_by_label=operator_map,
                )

                side_snapshots: Dict[str, HandSideSnapshot] = {}
                solve_futures: Dict[str, Future] = {}
                pending_meta: Dict[str, Tuple[TeleopHand, object]] = {}

                for side, hand in active_hands.items():
                    detection = detections.get(hand.mediapipe_label)
                    joint_pos = detection.joint_pos if detection is not None else None
                    keypoint_2d = (
                        detection.keypoint_2d if detection is not None else None
                    )
                    hand_detected = joint_pos is not None

                    if not hand_detected:
                        now = time.perf_counter()
                        if now - last_no_hand_warn[side] >= no_hand_warn_interval:
                            logger.warning(
                                f"{side} hand is not detected from Quest HTS."
                            )
                            last_no_hand_warn[side] = now
                        hand.command_robot(
                            None,
                            enable_interp=enable_interpolation_mode,
                            control_enabled=control_hand,
                        )
                        side_snapshots[side] = HandSideSnapshot(
                            keypoint_2d=keypoint_2d,
                            corrected_array=None,
                            qpos=None,
                            hand_detected=False,
                        )
                    else:
                        joint_pos = joint_pos[:, [2, 1, 0]]
                        pending_meta[side] = (hand, keypoint_2d)
                        solve_futures[side] = retarget_executor.submit(
                            hand.solve_from_joint_pos,
                            joint_pos,
                        )

                remaining = dict(solve_futures)
                while remaining:
                    done, _ = wait(
                        remaining.values(),
                        return_when=FIRST_COMPLETED,
                    )
                    for side, fut in list(remaining.items()):
                        if fut not in done:
                            continue
                        hand_meta, keypoint_2d = pending_meta[side]
                        try:
                            corrected_array, qpos = fut.result()
                        except Exception:
                            logger.exception(f"{side} hand retarget failed")
                            corrected_array, qpos = None, None
                        hand_meta.command_robot(
                            qpos,
                            enable_interp=enable_interpolation_mode,
                            control_enabled=control_hand,
                        )
                        side_snapshots[side] = HandSideSnapshot(
                            keypoint_2d=keypoint_2d,
                            corrected_array=corrected_array,
                            qpos=qpos,
                            hand_detected=True,
                        )
                        del remaining[side]

                shared_state.publish(
                    bgr=publish_bgr,
                    frame_counter=frame_counter,
                    hands=side_snapshots,
                )

        control_thread = threading.Thread(
            target=control_loop,
            daemon=True,
            name="teleop-control",
        )
        control_thread.start()

        try:
            run_quest_visualization_loop(
                visualizers, shared_state, teleop_stop, teleop_mode
            )
        finally:
            teleop_stop.set()
            control_thread.join(timeout=2.0)
            retarget_executor.shutdown(wait=False, cancel_futures=True)
    finally:
        quest_receiver.stop()
        close_all_visualizers(visualizers, render_process=render_process)
        stop_event.set()


def main(
    config_path: Tuple[str, ...] = (DEFAULT_CONFIG_PATH,),
    robot_dir: str = DEFAULT_ROBOT_DIR_PATH,
    hts_host: str = DEFAULT_HTS_HOST,
    hts_port: int = DEFAULT_HTS_PORT,
):
    """
    Retarget Quest HTS hand tracking (UDP) to one or two Sharpa Wave hands.

    In the HTS headset app, set IP to this PC's Wi-Fi address and port to
    ``hts_port`` (default 9000). Bind address ``hts_host`` is usually 0.0.0.0.
    """
    queue = multiprocessing.Queue(maxsize=4)
    stop_event = multiprocessing.Event()
    producer_process = multiprocessing.Process(
        target=produce_placeholder_frames,
        args=(queue, stop_event),
    )
    print(f"Quest HTS UDP listen {hts_host}:{hts_port}")
    consumer_process = multiprocessing.Process(
        target=start_retargeting,
        args=(queue, robot_dir, config_path, stop_event, hts_host, hts_port),
    )

    producer_process.start()
    consumer_process.start()

    try:
        consumer_process.join()
    finally:
        stop_event.set()
        producer_process.join(timeout=2.0)
        if producer_process.is_alive():
            producer_process.terminate()
            producer_process.join(timeout=1.0)
        print("Sharpa Wave Quest Teleop - Stop Hand Running Mode")
        SharpaWaveManager.get_instance().disconnect_all()
        print("Sharpa Wave Quest Teleop - Stopped")


if __name__ == "__main__":
    tyro.cli(main)
