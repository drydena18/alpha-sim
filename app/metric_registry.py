from __future__ import annotations

from typing import Callable, Dict, List
import numpy as np

from app.state_models import DataPacket, MetricResult

MetricFn = Callable[[DataPacket], MetricResult]

class MetricRegistry:
    """
    Registry for available GUI-selectable metrics.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, MetricFn] = {}

    def register(self, name: str, fn: MetricFn) -> None:
        self._metrics[name] = fn

    def names(self) -> List[str]:
        return list(self._metrics.keys())
    
    def compute(self, name: str, packet: DataPacket) -> MetricResult:
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' is not registered.")
        return self._metrics[name](packet)
    
def metric_kuramoto_r(packet: DataPacket) -> MetricResult:
    curve = np.cos(packet.phase_diff)
    sc = np.mean(np.cos(packet.phase_diff))
    ss = np.mean(np.sin(packet.phase_diff))
    scalar = float(np.sqrt(sc * sc + ss * ss))
    return MetricResult(curve = curve, scalar = scalar, name = "Kuramoto R", units = "R")

def metric_mean_phase_deg(packet: DataPacket) -> MetricResult:
    curve = np.degrees(packet.phase_diff)
    scalar = float(np.degrees(np.angle(np.mean(np.exp(1j * packet.phase_diff)))))
    return MetricResult(curve = curve, scalar = scalar, name = "Mean Phase Diff (deg)", units = "deg")

def metric_amplitude_balance(packet: DataPacket) -> MetricResult:
    a = np.abs(packet.signal_a)
    b = np.abs(packet.signal_b)
    curve = (a - b) / (a + b + 1e-8)
    scalar = float(np.mean(curve))
    return MetricResult(curve = curve, scalar = scalar, name = "Amplitude Balance", units = "a.u.")

def metric_amplitude_ratio(packet: DataPacket) -> MetricResult:
    a = np.abs(packet.signal_a)
    b = np.abs(packet.signal_b)
    curve = a / (b + 1e-8)
    scalar = float(np.mean(curve))
    return MetricResult(curve = curve, scalar = scalar, name = "Amplitude Ratio", units = "ratio")

def build_default_metric_registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register("Kuramoto R", metric_kuramoto_r)
    reg.register("Mean Phase Diff (deg)", metric_mean_phase_deg)
    reg.register("Amplitude Balance", metric_amplitude_balance)
    reg.register("Amplitude Ratio", metric_amplitude_ratio)
    return reg