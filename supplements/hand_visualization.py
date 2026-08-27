"""Visualization helpers for Sharpa Wave webcam teleop (OpenCV + Matplotlib wireframes)."""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing.context import BaseContext
from pathlib import Path
from queue import Empty, Full
from typing import Any, Dict, List, Literal, Optional, Tuple

# Matplotlib/OpenCV display backend (must be set before pyplot import).
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from single_hand_detector import HandDetector

HandSideName = Literal["left", "right"]

# ---------------------------------------------------------------------------
# Defaults — edit these if your machine layout differs from this repo.
# This module lives in supplements/; project root is one level up.
# ---------------------------------------------------------------------------
SUPPLEMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUPPLEMENTS_DIR.parent
DEFAULT_URDF_WAVE01 = REPO_ROOT / "sharpa-urdf-usd-xml" / "wave_01"

WEBCAM_WINDOW = "realtime_retargeting_demo"
MODE_WINDOW = "teleop_mode_selection"


def wireframe_window_name(side: HandSideName) -> str:
    return f"Debug View: Wireframes — {side.upper()}"


def default_urdf_path_for_side(side: HandSideName) -> str:
    return str(DEFAULT_URDF_WAVE01 / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_with_wrist.urdf")


DEFAULT_URDF_PATH = default_urdf_path_for_side("left")


def mp_to_frame_names(side: HandSideName) -> list[str]:
    prefix = f"{side}_"
    return [
        f"{prefix}hand_C_MC",
        f"{prefix}thumb_MC",
        f"{prefix}thumb_MCP_FE",
        f"{prefix}thumb_IP",
        f"{prefix}thumb_fingertip",
        f"{prefix}index_MCP_FE",
        f"{prefix}index_PIP",
        f"{prefix}index_DIP",
        f"{prefix}index_fingertip",
        f"{prefix}middle_MCP_FE",
        f"{prefix}middle_PIP",
        f"{prefix}middle_DIP",
        f"{prefix}middle_fingertip",
        f"{prefix}ring_MCP_FE",
        f"{prefix}ring_PIP",
        f"{prefix}ring_DIP",
        f"{prefix}ring_fingertip",
        f"{prefix}pinky_MCP_FE",
        f"{prefix}pinky_PIP",
        f"{prefix}pinky_DIP",
        f"{prefix}pinky_fingertip",
    ]


# Standard 21-point MediaPipe hand skeletal structure.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

MP_TO_FRAME = mp_to_frame_names("left")


class TeleopMode(Enum):
    BOTH = "both"
    VISUALIZATION_ONLY = "visualization_only"
    HAND_ONLY = "hand_only"


_MODE_KEYS = {
    ord("1"): TeleopMode.BOTH,
    ord("2"): TeleopMode.VISUALIZATION_ONLY,
    ord("3"): TeleopMode.HAND_ONLY,
}


def teleop_mode_label(mode: TeleopMode) -> str:
    if mode == TeleopMode.BOTH:
        return "wireframe graphs + hand control"
    if mode == TeleopMode.VISUALIZATION_ONLY:
        return "wireframe graphs only (no robot motion)"
    return "hand control only (no wireframe graphs)"


@dataclass
class HandSideSnapshot:
    keypoint_2d: Any
    corrected_array: Optional[np.ndarray]
    qpos: Optional[np.ndarray]
    hand_detected: bool


@dataclass
class TeleopSnapshot:
    bgr: np.ndarray
    frame_counter: int
    hands: Dict[str, HandSideSnapshot]
    camera_fps: Optional[float] = None


class SharedTeleopState:
    """Thread-safe latest frame + per-side retargeting results for the viz thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: Optional[TeleopSnapshot] = None

    def publish(
        self,
        *,
        bgr: np.ndarray,
        frame_counter: int,
        hands: Dict[str, HandSideSnapshot],
        camera_fps: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._snapshot = TeleopSnapshot(
                bgr=bgr.copy(),
                frame_counter=frame_counter,
                hands={
                    side: HandSideSnapshot(
                        keypoint_2d=side_snap.keypoint_2d,
                        corrected_array=(
                            side_snap.corrected_array.copy()
                            if side_snap.corrected_array is not None
                            else None
                        ),
                        qpos=(
                            side_snap.qpos.copy()
                            if side_snap.qpos is not None
                            else None
                        ),
                        hand_detected=side_snap.hand_detected,
                    )
                    for side, side_snap in hands.items()
                },
                camera_fps=camera_fps,
            )

    def get_latest(self) -> Optional[TeleopSnapshot]:
        with self._lock:
            return self._snapshot


def _draw_camera_fps_overlay(image: np.ndarray, camera_fps: Optional[float]) -> np.ndarray:
    """Draw live laptop-camera FPS in the top-left of the webcam window."""
    if camera_fps is None:
        return image
    label = f"Camera FPS: {camera_fps:.1f}"
    origin = (10, 28)
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _render_side_wireframe_bgr(
    *,
    side: str,
    rotated_array: np.ndarray,
    qpos: np.ndarray,
    model,
    data,
    fig,
    ax_mp,
    ax_robot,
    mp_to_frame: list[str],
) -> np.ndarray:
    import pinocchio as pin

    pin.forwardKinematics(model, data, qpos)
    pin.updateFramePlacements(model, data)

    robot_points = []
    for name in mp_to_frame:
        frame_id = model.getFrameId(name)
        robot_points.append(data.oMf[frame_id].translation)
    robot_points = np.array(robot_points)
    robot_centered = robot_points - robot_points[0]

    ax_mp.cla()
    ax_robot.cla()
    for axis in (ax_mp, ax_robot):
        axis.set_xlim([-0.2, 0.2])
        axis.set_ylim([-0.2, 0.2])
        axis.set_zlim([-0.2, 0.2])
        axis.set_xlabel("X (Forward)")
        axis.set_ylabel("Y (Left/Right)")
        axis.set_zlabel("Z (Up/Down)")

    ax_mp.set_title(f"MediaPipe Target ({side.upper()})")
    ax_robot.set_title(f"Pinocchio Reality ({side.upper()})")
    ax_mp.scatter(
        rotated_array[:, 0],
        rotated_array[:, 1],
        rotated_array[:, 2],
        c="blue",
        s=20,
    )
    for bone in HAND_BONES:
        pt1, pt2 = bone[0], bone[1]
        ax_mp.plot(
            [rotated_array[pt1, 0], rotated_array[pt2, 0]],
            [rotated_array[pt1, 1], rotated_array[pt2, 1]],
            [rotated_array[pt1, 2], rotated_array[pt2, 2]],
            c="blue",
            linewidth=2,
        )

    ax_robot.scatter(
        robot_centered[:, 0],
        robot_centered[:, 1],
        robot_centered[:, 2],
        c="red",
        s=20,
    )
    for bone in HAND_BONES:
        pt1, pt2 = bone[0], bone[1]
        try:
            ax_robot.plot(
                [robot_centered[pt1, 0], robot_centered[pt2, 0]],
                [robot_centered[pt1, 1], robot_centered[pt2, 1]],
                [robot_centered[pt1, 2], robot_centered[pt2, 2]],
                c="red",
                linewidth=2,
            )
        except IndexError:
            pass

    for axis in (ax_mp, ax_robot):
        axis.margins(0)

    fig.canvas.draw()
    plot_img = np.array(fig.canvas.buffer_rgba())
    return cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)


def _wireframe_render_main(
    job_q: Any,
    result_q: Any,
    side_configs: List[Tuple[str, str]],
) -> None:
    """Child process entry: own matplotlib/Pinocchio; never share GIL with teleop."""
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pinocchio as pin

    side_state: Dict[str, Dict[str, Any]] = {}
    for side, urdf_path in side_configs:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        fig = plt.figure(figsize=(8, 4), dpi=80)
        ax_mp = fig.add_subplot(121, projection="3d")
        ax_robot = fig.add_subplot(122, projection="3d")
        for axis in (ax_mp, ax_robot):
            axis.set_xlim([-0.2, 0.2])
            axis.set_ylim([-0.2, 0.2])
            axis.set_zlim([-0.2, 0.2])
            axis.set_xlabel("X (Forward)")
            axis.set_ylabel("Y (Left/Right)")
            axis.set_zlabel("Z (Up/Down)")
        side_state[side] = {
            "model": model,
            "data": data,
            "fig": fig,
            "ax_mp": ax_mp,
            "ax_robot": ax_robot,
            "mp_to_frame": mp_to_frame_names(side),  # type: ignore[arg-type]
        }

    try:
        while True:
            job = job_q.get()
            if job is None:
                break
            hands = job.get("hands") or {}
            images: Dict[str, np.ndarray] = {}
            for side, payload in hands.items():
                state = side_state.get(side)
                if state is None:
                    continue
                corrected_array, qpos = payload
                images[side] = _render_side_wireframe_bgr(
                    side=side,
                    rotated_array=corrected_array,
                    qpos=qpos,
                    model=state["model"],
                    data=state["data"],
                    fig=state["fig"],
                    ax_mp=state["ax_mp"],
                    ax_robot=state["ax_robot"],
                    mp_to_frame=state["mp_to_frame"],
                )
            result = {
                "frame_counter": job.get("frame_counter", -1),
                "images": images,
            }
            try:
                result_q.put_nowait(result)
            except Full:
                try:
                    result_q.get_nowait()
                except Empty:
                    pass
                try:
                    result_q.put_nowait(result)
                except Full:
                    pass
    finally:
        for state in side_state.values():
            try:
                plt.close(state["fig"])
            except Exception:
                pass


class WireframeRenderProcess:
    """Spawn-process backend shared by one or more HandVisualizer instances."""

    def __init__(
        self,
        side_configs: List[Tuple[str, str]],
        *,
        wireframe_interval: int = 5,
    ) -> None:
        if not side_configs:
            raise ValueError("side_configs must contain at least one (side, urdf_path)")
        self.side_configs = list(side_configs)
        self.wireframe_interval = wireframe_interval
        self._ctx: BaseContext = multiprocessing.get_context("spawn")
        self._job_q: Any = None
        self._result_q: Any = None
        self._proc: Optional[multiprocessing.Process] = None
        self._latest: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self) -> None:
        if self.started:
            return
        self._job_q = self._ctx.Queue(maxsize=1)
        self._result_q = self._ctx.Queue(maxsize=2)
        self._proc = self._ctx.Process(
            target=_wireframe_render_main,
            args=(self._job_q, self._result_q, self.side_configs),
            name="wireframe-render",
            daemon=True,
        )
        self._proc.start()

    def submit(
        self,
        hands: Dict[str, Tuple[np.ndarray, np.ndarray]],
        frame_counter: int,
    ) -> None:
        """Submit one or more hands together for the same frame (no stagger)."""
        if (
            not hands
            or self._job_q is None
            or frame_counter % self.wireframe_interval != 0
        ):
            return
        job = {
            "frame_counter": frame_counter,
            "hands": {
                side: (corrected.copy(), qpos.copy())
                for side, (corrected, qpos) in hands.items()
            },
        }
        try:
            self._job_q.put_nowait(job)
        except Full:
            try:
                self._job_q.get_nowait()
            except Empty:
                pass
            try:
                self._job_q.put_nowait(job)
            except Full:
                pass

    def _drain_results(self) -> None:
        if self._result_q is None:
            return
        while True:
            try:
                result = self._result_q.get_nowait()
            except Empty:
                break
            images = result.get("images") or {}
            with self._lock:
                for side, bgr in images.items():
                    self._latest[side] = bgr

    def poll(self, side: str) -> Optional[np.ndarray]:
        self._drain_results()
        with self._lock:
            return self._latest.get(side)

    def stop(self) -> None:
        proc = self._proc
        job_q = self._job_q
        self._proc = None
        if job_q is not None:
            try:
                job_q.put_nowait(None)
            except Full:
                try:
                    job_q.get_nowait()
                except Empty:
                    pass
                try:
                    job_q.put_nowait(None)
                except Full:
                    pass
            except Exception:
                pass
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self._job_q = None
        self._result_q = None


def run_visualization_loop(
    visualizers: Dict[str, "HandVisualizer"],
    shared_state: SharedTeleopState,
    stop_event: threading.Event,
) -> None:
    """Display webcam + optional per-side wireframes on the main thread."""
    last_wireframe_frame = -1
    show_wireframes = any(v.show_wireframes for v in visualizers.values())
    render_process = next(
        (v.render_process for v in visualizers.values() if v.render_process is not None),
        None,
    )
    wireframe_interval = (
        render_process.wireframe_interval
        if render_process is not None
        else 5
    )

    while not stop_event.is_set():
        snapshot = shared_state.get_latest()
        if snapshot is not None:
            display_bgr = snapshot.bgr.copy()
            for side_snap in snapshot.hands.values():
                if side_snap.keypoint_2d is not None:
                    display_bgr = HandDetector.draw_skeleton_on_image(
                        display_bgr,
                        side_snap.keypoint_2d,
                        style="default",
                    )
            display_bgr = _draw_camera_fps_overlay(display_bgr, snapshot.camera_fps)
            cv2.imshow(WEBCAM_WINDOW, display_bgr)

            if show_wireframes and render_process is not None:
                batch: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                if (
                    snapshot.frame_counter % wireframe_interval == 0
                    and snapshot.frame_counter != last_wireframe_frame
                ):
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
                            batch[side] = (
                                side_snap.corrected_array,
                                side_snap.qpos,
                            )
                    if batch:
                        # Bimanual: all detected hands in one job (no stagger).
                        render_process.submit(batch, snapshot.frame_counter)
                        last_wireframe_frame = snapshot.frame_counter

                for side, visualizer in visualizers.items():
                    wireframe_bgr = visualizer.poll_wireframe_image()
                    if wireframe_bgr is not None:
                        cv2.imshow(visualizer.wireframe_window, wireframe_bgr)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            stop_event.set()
            break

        time.sleep(0.001 if show_wireframes else 0.0)


def _draw_mode_overlay(image: np.ndarray, *, selected: Optional[TeleopMode] = None) -> np.ndarray:
    overlay = image.copy()
    h, w = overlay.shape[:2]
    panel_h = 200
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, overlay)

    cv2.putText(
        overlay,
        "Calibration complete - select teleop mode:",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    options = [
        ("1", TeleopMode.BOTH, "Wireframe graphs + hand control"),
        ("2", TeleopMode.VISUALIZATION_ONLY, "Wireframe graphs only (no robot motion)"),
        ("3", TeleopMode.HAND_ONLY, "Hand control only (no wireframe graphs)"),
    ]
    y = 68
    for key, mode, label in options:
        color = (0, 220, 0) if selected == mode else (220, 220, 220)
        cv2.putText(
            overlay,
            f"[{key}] {label}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2 if selected == mode else 1,
            cv2.LINE_AA,
        )
        y += 30

    if selected is not None:
        cv2.putText(
            overlay,
            f"Selected: {teleop_mode_label(selected)}. Starting teleop...",
            (20, panel_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            overlay,
            "Press 1, 2, or 3. Press q to quit.",
            (20, panel_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return overlay


def prompt_teleop_mode(
    bgr: np.ndarray,
    window_name: str = MODE_WINDOW,
) -> TeleopMode:
    """Show mode selection UI after calibration; blocks until the user chooses."""
    while True:
        frame = _draw_mode_overlay(bgr)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in _MODE_KEYS:
            mode = _MODE_KEYS[key]
            cv2.imshow(window_name, _draw_mode_overlay(bgr, selected=mode))
            cv2.waitKey(800)
            cv2.destroyWindow(window_name)
            return mode
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Teleop mode selection aborted by user.")


class HandVisualizer:
    """Optional wireframe debug view for one hand side (render via shared process)."""

    def __init__(
        self,
        mode: TeleopMode,
        side: HandSideName = "left",
        urdf_path: Optional[str] = None,
        wireframe_interval: int = 5,
        render_process: Optional[WireframeRenderProcess] = None,
    ):
        self.mode = mode
        self.side = side
        self.wireframe_interval = wireframe_interval
        self.wireframe_window = wireframe_window_name(side)
        self.mp_to_frame = mp_to_frame_names(side)
        self.show_wireframes = mode in (TeleopMode.BOTH, TeleopMode.VISUALIZATION_ONLY)
        self.render_process = render_process

        if urdf_path is None:
            urdf_path = default_urdf_path_for_side(side)
        self.urdf_path = urdf_path

        if self.show_wireframes:
            cv2.namedWindow(self.wireframe_window, cv2.WINDOW_NORMAL)

    def submit_wireframe(
        self,
        corrected_array: np.ndarray,
        qpos: np.ndarray,
        frame_counter: int,
    ) -> None:
        """Legacy single-hand submit; prefer batched submit via run_visualization_loop."""
        if not self.show_wireframes or self.render_process is None:
            return
        self.render_process.submit(
            {self.side: (corrected_array, qpos)},
            frame_counter,
        )

    def poll_wireframe_image(self) -> Optional[np.ndarray]:
        if self.render_process is None:
            return None
        return self.render_process.poll(self.side)

    def close(self) -> None:
        # Shared WireframeRenderProcess is stopped by the entrypoint / close helper.
        try:
            cv2.destroyWindow(self.wireframe_window)
        except cv2.error:
            pass


def close_all_visualizers(
    visualizers: Dict[str, HandVisualizer],
    render_process: Optional[WireframeRenderProcess] = None,
) -> None:
    for visualizer in visualizers.values():
        visualizer.close()
    if render_process is not None:
        render_process.stop()
    else:
        # Fallback: stop any process still referenced by a visualizer.
        seen: set[int] = set()
        for visualizer in visualizers.values():
            proc = visualizer.render_process
            if proc is not None and id(proc) not in seen:
                seen.add(id(proc))
                proc.stop()
    try:
        cv2.destroyWindow(WEBCAM_WINDOW)
        cv2.destroyWindow(MODE_WINDOW)
    except cv2.error:
        pass
    cv2.destroyAllWindows()
