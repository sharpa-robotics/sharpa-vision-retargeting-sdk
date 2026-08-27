"""Webcam teleop for Sharpa Wave hands (MediaPipe → landmark scale → retarget).

Frozen palm-span and per-finger MCP→tip scales are computed once after
calibration from neutral landmarks versus URDF lengths. Each control frame
corrects landmarks, scales them for retargeting, and shows post-calibration
landmarks on the wireframe.
"""

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
DEFAULT_CAMERA_PATH = "/dev/video0"

if str(SUPPLEMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPLEMENTS_DIR))
if DEFAULT_SHARPA_SDK_PYTHON not in sys.path:
    sys.path.insert(0, DEFAULT_SHARPA_SDK_PYTHON)

import multiprocessing
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from queue import Empty, Full
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tyro
from dex_retargeting.retargeting_config import RetargetingConfig
from frame_queue_utils import flush_queue, get_latest_frame
from hand_calibration import run_calibration
from hand_visualization import (
    HandSideSnapshot,
    HandVisualizer,
    SharedTeleopState,
    TeleopMode,
    WireframeRenderProcess,
    close_all_visualizers,
    prompt_teleop_mode,
    run_visualization_loop,
    teleop_mode_label,
)
from loguru import logger
from sharpa import SharpaWaveManager
from single_hand_detector import HandDetector
from teleop_hand import (
    TeleopHand,
    build_teleop_hand,
    connect_wave_hands,
    initialize_wave,
    mediapipe_label_for_side,
    operator2mano_for_side,
    parse_config_paths,
    resolve_robot_dir,
)

print("Sharpa Wave Webcam Teleop - Starting")


def _fetch_bgr_frame(queue: multiprocessing.Queue) -> Tuple[Optional[float], np.ndarray, int]:
    """
    Gets the latest frame from the webcam
    """
    try:
        return get_latest_frame(queue, timeout=5.0)
    except Empty as exc:
        raise RuntimeError(
            "Fail to fetch image from camera in 5 secs. Please check your web camera device."
        ) from exc


def start_retargeting(
    queue: multiprocessing.Queue,
    robot_dir: str,
    config_paths: Tuple[str, ...],
    stop_event: multiprocessing.Event,
    camera_fps_value: Optional["multiprocessing.Value"] = None,):
    """
    Initializes the hands and executes calibration stage and teleop mode selection
    """
    visualizers: Dict[str, HandVisualizer] = {}
    teleop_stop = threading.Event()
    render_process = None
    try:
        configs = parse_config_paths(config_paths)
        expected_sides = list(configs.keys())
        logger.info(f"Requested teleop sides: {expected_sides}")

        detector = HandDetector(num_hands=len(expected_sides))
        operator_map = {
            mediapipe_label_for_side(side): operator2mano_for_side(side)
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

            #ensures low latency by flushing old webcam frames
            flush_queue(queue)
            calibration = run_calibration(
                detector,
                queue,
                side=side,
                retargeting_type=retargeting.optimizer.retargeting_type,
                target_link_human_indices=retargeting.optimizer.target_link_human_indices,
            )

            #handles scaling and configuration transformations
            active_hands[side] = build_teleop_hand(
                side,
                config_path,
                robot_dir,
                wave=wave,
                device_sn=device_sn,
                calibration=calibration,
            )
            logger.info(
                f"Active {side} hand"
                + (f" (SN {device_sn})" if device_sn else " (no device)")
            )

        if not active_hands:
            raise RuntimeError("No teleop hands could be initialized.")

        flush_queue(queue)
        _, mode_frame, _ = _fetch_bgr_frame(queue)
        teleop_mode = prompt_teleop_mode(mode_frame)
        logger.info(f"Teleop mode selected: {teleop_mode_label(teleop_mode)}")

        #initializes relevant processes depending on selected TeleopMode
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

        def control_loop() -> None:
            """
            Main retarteting loop, passing the obtained frame to MediaPipe, processing its landmark data, sending to the retargeter, and publishing as necessary
            """
            nonlocal frame_counter
            while not teleop_stop.is_set():
                frame_counter += 1
                _, bgr, _ = _fetch_bgr_frame(queue)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.time() * 1000)
                detections = detector.detect_all(
                    rgb,
                    timestamp_ms,
                    operator2mano_by_label=operator_map,
                )

                side_snapshots: Dict[str, HandSideSnapshot] = {}
                solve_futures: Dict[str, Future] = {}
                pending_meta: Dict[str, Tuple[TeleopHand, object]] = {}

                for side, hand in active_hands.items():
                    detection = detections.get(hand.mediapipe_label)
                    joint_pos = detection.joint_pos if detection is not None else None
                    keypoint_2d = detection.keypoint_2d if detection is not None else None
                    hand_detected = joint_pos is not None

                    if not hand_detected:
                        now = time.perf_counter()
                        if now - last_no_hand_warn[side] >= no_hand_warn_interval:
                            logger.warning(f"{side} hand is not detected.")
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
                    bgr=bgr,
                    frame_counter=frame_counter,
                    hands=side_snapshots,
                    camera_fps=(
                        float(camera_fps_value.value)
                        if camera_fps_value is not None
                        else None
                    ),
                )

        control_thread = threading.Thread(
            target=control_loop,
            daemon=True,
            name="teleop-control",
        )
        control_thread.start()

        try:
            run_visualization_loop(visualizers, shared_state, teleop_stop)
        finally:
            teleop_stop.set()
            control_thread.join(timeout=2.0)
            retarget_executor.shutdown(wait=False, cancel_futures=True)
    finally:
        close_all_visualizers(visualizers, render_process=render_process)
        stop_event.set()


def produce_frame(
    queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    camera_path: Optional[str] = None,
    camera_fps_value: Optional["multiprocessing.Value"] = None,
):
    """
    Display webcam with overlayed skeleton and fps tracker
    """
    if camera_path is None:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)
    ema_fps = 0.0
    last_grab_t = time.perf_counter()
    try:
        while cap.isOpened() and not stop_event.is_set():
            success, image = cap.read()
            if not success:
                continue
            now = time.perf_counter()
            dt = now - last_grab_t
            last_grab_t = now
            if dt > 1e-6:
                inst_fps = 1.0 / dt
                ema_fps = inst_fps if ema_fps <= 0.0 else (0.85 * ema_fps + 0.15 * inst_fps)
                if camera_fps_value is not None:
                    camera_fps_value.value = float(ema_fps)
            image = cv2.flip(image, 1)
            item = (time.time(), image)
            try:
                queue.put_nowait(item)
            except Full:
                try:
                    queue.get_nowait()
                    queue.put_nowait(item)
                except (Empty, Full):
                    pass
    finally:
        cap.release()


def main(
    config_path: Tuple[str, ...] = (DEFAULT_CONFIG_PATH,),
    robot_dir: str = DEFAULT_ROBOT_DIR_PATH,
    camera_path: str = DEFAULT_CAMERA_PATH,
):
    """Webcam teleop: MediaPipe landmarks → palm/finger scales → Sharpa Wave."""
    queue = multiprocessing.Queue(maxsize=4)
    stop_event = multiprocessing.Event()
    camera_fps_value = multiprocessing.Value("d", 0.0)
    producer_process = multiprocessing.Process(
        target=produce_frame,
        args=(queue, stop_event, camera_path, camera_fps_value),
    )
    print("Camera On")
    consumer_process = multiprocessing.Process(
        target=start_retargeting,
        args=(queue, robot_dir, config_path, stop_event, camera_fps_value),
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
        print("Sharpa Wave Webcam - Stop Hand Running Mode")
        SharpaWaveManager.get_instance().disconnect_all()
        print("Sharpa Wave Webcam - Stopped")


if __name__ == "__main__":
    tyro.cli(main)
