from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class DataPacket:
    """
    Standard packet sent to the GUI for plotting.
    """
    signal_a: np.ndarray
    signal_b: np.ndarray
    phase_diff: np.ndarray
    time_axis: np.ndarray
    time_seconds: float
    meta: Dict[str, float] = field(default_factory = dict)

@dataclass
class MetricResult:
    """
    Standard output from a metric function.
    """
    curve: np.ndarray
    scalar: float
    name: str
    units: str = "a.u."

@dataclass
class AppConfig:
    """
    GUI/runtime config values.
    """
    buf_len: int = 1000
    dt: float = 1.0 / 1000.0
    steps_per_frame: int = 6
    timer_interval_ms: int = 16
    lib_path: Optional[str] = None

@dataclass
class SourceStatus:
    """
    Lightweight status object for current source/mode.
    """
    mode_name: str
    status_text: str
    loaded_file: Optional[str] = None