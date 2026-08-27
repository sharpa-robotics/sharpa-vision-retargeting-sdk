"""Utilities for low-latency frame queue handling."""

from __future__ import annotations

from queue import Empty
from typing import Optional, Tuple, Union

import numpy as np


def unwrap_frame(
    item: Union[np.ndarray, Tuple[float, np.ndarray]],
) -> Tuple[Optional[float], np.ndarray]:
    if isinstance(item, tuple) and len(item) == 2:
        return float(item[0]), item[1]
    return None, item


def flush_queue(queue) -> int:
    """Discard all queued frames; returns number drained."""
    drained = 0
    while True:
        try:
            queue.get_nowait()
            drained += 1
        except Empty:
            break
    return drained


def get_latest_frame(
    queue,
    *,
    timeout: float = 5.0,
) -> Tuple[Optional[float], np.ndarray, int]:
    """Return the newest queued frame, dropping any older buffered frames."""
    item = None
    drained = 0

    try:
        item = queue.get_nowait()
        drained = 1
    except Empty:
        pass

    while True:
        try:
            item = queue.get_nowait()
            drained += 1
        except Empty:
            break

    if item is None:
        item = queue.get(timeout=timeout)
        drained = 0

    enqueue_ts, frame = unwrap_frame(item)
    return enqueue_ts, frame, drained
