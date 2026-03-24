from __future__ import annotations

from typing import Optional

from app.metric_registry import MetricRegistry
from app.state_models import DataPacket, MetricResult


class AppController:
    """
    Thin orchestration layer between:
    - current data source
    - selected metric
    - GUI plotting
    """

    def __init__(self, metric_registry: MetricRegistry):
        self.metric_registry = metric_registry
        self.current_source = None
        self.current_metric_name: Optional[str] = None

    def set_source(self, source) -> None:
        self.current_source = source

    def set_metric(self, metric_name: str) -> None:
        self.current_metric_name = metric_name

    def get_current_packet(self) -> DataPacket:
        if self.current_source is None:
            raise RuntimeError("No current source has been set.")
        return self.current_source.get_packet()

    def compute_metric(self, packet: DataPacket) -> MetricResult:
        if self.current_metric_name is None:
            raise RuntimeError("No current metric has been set.")
        return self.metric_registry.compute(self.current_metric_name, packet)

    def reset_source(self) -> None:
        if self.current_source is not None:
            self.current_source.reset()

    def apply_params(self, params: dict) -> None:
        if self.current_source is not None and hasattr(self.current_source, "apply_params"):
            self.current_source.apply_params(params)