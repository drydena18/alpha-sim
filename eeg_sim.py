"""
EEG Alpha Interaction Sandbox / Replay App

Purpose
-------
This app is designed to support two modes:
1. Demo mode: two coupled van der Pol oscillators driven by a C core.
2. EEG replay mode: a future time-resolved replay pipeline for real EEG-derived
   alpha metrics.

The EEG replay pieces are scaffolded but intentionally not yet fully functional.
They provide a clean place to bolt in MNE / EEGLAB-loading, band-pass filtering,
Hilbert-based envelopes, ROI aggregation, and one or more alpha interaction
metrics once those are scientifically defined.

Dependencies
------------
Demo mode:
  pip install pyqtgraph PyQt5 numpy

Future EEG replay mode:
  pip install mne scipy

Build the C library first:
  gcc -O2 -march=native -shared -fPIC -o vdp_core.so vdp_core_v2.c -lm

Run:
  python eeg_sim_v2.py
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


# =============================================================================
# Configuration
# =============================================================================

HERE = pathlib.Path(__file__).resolve().parent
LIB_PATH = HERE / "vdp_core.so"

BUF_LEN = 1000
DT = 1.0 / 1000.0
STEPS_PER_FRAME = 6
TIMER_INTERVAL_MS = 16

BLUE = (55, 138, 221)
GREEN = (29, 158, 117)
PURPLE = (127, 119, 221)
ORANGE = (220, 143, 63)
GRAY = (160, 160, 160)

DEFAULT_PARAMS: Dict[str, float] = {
    "freq_a": 10.0,
    "freq_b": 9.0,
    "mu_a": 0.30,
    "mu_b": 0.30,
    "amp_a": 1.00,
    "amp_b": 1.00,
    "coupling": 0.05,
    "noise": 0.05,
    "alertness_a": 0.50,
    "alertness_b": 0.50,
}

PARAM_SPECS = [
    ("freq_a", "Freq A (Hz)", 7.0, 13.0, 10),
    ("mu_a", "Damping A (μ)", 0.05, 2.0, 100),
    ("amp_a", "Amplitude A", 0.2, 3.0, 10),
    ("alertness_a", "Alertness A", 0.0, 1.0, 100),
    ("freq_b", "Freq B (Hz)", 7.0, 13.0, 10),
    ("mu_b", "Damping B (μ)", 0.05, 2.0, 100),
    ("amp_b", "Amplitude B", 0.2, 3.0, 10),
    ("alertness_b", "Alertness B", 0.0, 1.0, 100),
    ("coupling", "Coupling K", 0.0, 0.5, 100),
    ("noise", "Noise σ", 0.0, 0.4, 100),
]


# =============================================================================
# ctypes bridge
# =============================================================================

_c_double_p = ctypes.POINTER(ctypes.c_double)


class SimCore:
    def __init__(self, lib_path: pathlib.Path, buf_len: int):
        if not lib_path.exists():
            sys.exit(
                "vdp_core.so not found. Build it first, e.g.:\n"
                "  gcc -O2 -march=native -shared -fPIC -o vdp_core.so vdp_core_v2.c -lm"
            )

        self.lib = ctypes.CDLL(str(lib_path))
        self._configure_signatures()

        self.buf_len = buf_len
        self.ctx = self.lib.sim_create(buf_len)
        if not self.ctx:
            sys.exit("Failed to create simulation context from vdp_core.so")

    def _configure_signatures(self) -> None:
        lib = self.lib

        lib.sim_create.restype = ctypes.c_void_p
        lib.sim_create.argtypes = [ctypes.c_int]

        lib.sim_free.restype = None
        lib.sim_free.argtypes = [ctypes.c_void_p]

        lib.sim_reset.restype = None
        lib.sim_reset.argtypes = [ctypes.c_void_p]

        lib.sim_reset_with_seed.restype = None
        lib.sim_reset_with_seed.argtypes = [ctypes.c_void_p]

        lib.sim_advance.restype = None
        lib.sim_advance.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]

        lib.sim_get_buffers.restype = None
        lib.sim_get_buffers.argtypes = [
            ctypes.c_void_p,
            _c_double_p,
            _c_double_p,
            _c_double_p,
            ctypes.c_int,
        ]

        for getter in [
            "sim_sync_index",
            "sim_get_time",
            "sim_get_phase_diff",
            "sim_get_freq_a",
            "sim_get_freq_b",
            "sim_get_alertness_a",
            "sim_get_alertness_b",
            "sim_get_effective_freq_a",
            "sim_get_effective_freq_b",
        ]:
            getattr(lib, getter).restype = ctypes.c_double
            getattr(lib, getter).argtypes = [ctypes.c_void_p]

        lib.sim_get_buffer_len.restype = ctypes.c_int
        lib.sim_get_buffer_len.argtypes = [ctypes.c_void_p]

        for name in [
            "sim_set_freq_a",
            "sim_set_freq_b",
            "sim_set_mu_a",
            "sim_set_mu_b",
            "sim_set_amp_a",
            "sim_set_amp_b",
            "sim_set_coupling",
            "sim_set_noise",
            "sim_set_alertness_a",
            "sim_set_alertness_b",
        ]:
            getattr(lib, name).restype = None
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_double]

    def free(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.sim_free(self.ctx)
            self.ctx = None

    def apply_params(self, params: Dict[str, float]) -> None:
        for key, value in params.items():
            setter = getattr(self.lib, f"sim_set_{key}", None)
            if setter is not None:
                setter(self.ctx, float(value))

    def reset(self, exact_seed_reset: bool = False) -> None:
        if exact_seed_reset:
            self.lib.sim_reset_with_seed(self.ctx)
        else:
            self.lib.sim_reset(self.ctx)

    def advance(self, steps: int, dt: float) -> None:
        self.lib.sim_advance(self.ctx, steps, dt)

    def get_buffers(self, out_a: np.ndarray, out_b: np.ndarray, out_phase: np.ndarray) -> None:
        if len(out_a) < self.buf_len or len(out_b) < self.buf_len or len(out_phase) < self.buf_len:
            raise ValueError("Output arrays must be at least buf_len long.")
        self.lib.sim_get_buffers(
            self.ctx,
            out_a.ctypes.data_as(_c_double_p),
            out_b.ctypes.data_as(_c_double_p),
            out_phase.ctypes.data_as(_c_double_p),
            self.buf_len,
        )

    def get_sync(self) -> float:
        return float(self.lib.sim_sync_index(self.ctx))

    def get_time(self) -> float:
        return float(self.lib.sim_get_time(self.ctx))

    def get_phase_diff(self) -> float:
        return float(self.lib.sim_get_phase_diff(self.ctx))

    def get_effective_freqs(self) -> tuple[float, float]:
        fa = float(self.lib.sim_get_effective_freq_a(self.ctx))
        fb = float(self.lib.sim_get_effective_freq_b(self.ctx))
        return fa, fb


# =============================================================================
# Data packet abstraction
# =============================================================================

@dataclass
class DataPacket:
    signal_a: np.ndarray
    signal_b: np.ndarray
    phase_diff: np.ndarray
    time_axis: np.ndarray
    time_seconds: float
    meta: Dict[str, float]


# =============================================================================
# Metric definitions
# =============================================================================

class MetricRegistry:
    """
    Registry for scalar and time-resolved alpha interaction metrics.

    The point of this registry is not that all metrics are valid today. It is to
    make swapping in your final scientifically-defensible metric trivial later.
    """

    def __init__(self):
        self._metrics: Dict[str, Callable[[DataPacket], tuple[np.ndarray, float]]] = {
            "Kuramoto R": self.metric_kuramoto_r,
            "Mean phase diff (deg)": self.metric_mean_phase_deg,
            "Amplitude balance": self.metric_amplitude_balance,
            "Amplitude ratio": self.metric_amplitude_ratio,
        }

    @property
    def names(self) -> List[str]:
        return list(self._metrics.keys())

    def compute(self, metric_name: str, packet: DataPacket) -> tuple[np.ndarray, float]:
        fn = self._metrics[metric_name]
        return fn(packet)

    @staticmethod
    def metric_kuramoto_r(packet: DataPacket) -> tuple[np.ndarray, float]:
        # Time-resolved curve is cos(phase diff); scalar is classic R.
        curve = np.cos(packet.phase_diff)
        sc = np.mean(np.cos(packet.phase_diff))
        ss = np.mean(np.sin(packet.phase_diff))
        scalar = float(np.sqrt(sc * sc + ss * ss))
        return curve, scalar

    @staticmethod
    def metric_mean_phase_deg(packet: DataPacket) -> tuple[np.ndarray, float]:
        curve = np.degrees(packet.phase_diff)
        scalar = float(np.degrees(np.angle(np.mean(np.exp(1j * packet.phase_diff)))))
        return curve, scalar

    @staticmethod
    def metric_amplitude_balance(packet: DataPacket) -> tuple[np.ndarray, float]:
        a = np.abs(packet.signal_a)
        b = np.abs(packet.signal_b)
        curve = (a - b) / (a + b + 1e-8)
        return curve, float(np.mean(curve))

    @staticmethod
    def metric_amplitude_ratio(packet: DataPacket) -> tuple[np.ndarray, float]:
        a = np.abs(packet.signal_a)
        b = np.abs(packet.signal_b)
        curve = a / (b + 1e-8)
        return curve, float(np.mean(curve))


# =============================================================================
# Data sources
# =============================================================================

class BaseSource:
    mode_name = "Base"

    def get_packet(self) -> DataPacket:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def apply_params(self, params: Dict[str, float]) -> None:
        pass


class DemoSource(BaseSource):
    mode_name = "Demo"

    def __init__(self, core: SimCore, buf_len: int, dt: float, steps_per_frame: int):
        self.core = core
        self.buf_len = buf_len
        self.dt = dt
        self.steps_per_frame = steps_per_frame

        self.arr_a = np.zeros(buf_len, dtype=np.float64)
        self.arr_b = np.zeros(buf_len, dtype=np.float64)
        self.arr_phase = np.zeros(buf_len, dtype=np.float64)
        self.time_axis = np.linspace(-(buf_len - 1) * dt, 0.0, buf_len)

    def get_packet(self) -> DataPacket:
        self.core.advance(self.steps_per_frame, self.dt)
        self.core.get_buffers(self.arr_a, self.arr_b, self.arr_phase)
        fa, fb = self.core.get_effective_freqs()
        return DataPacket(
            signal_a=self.arr_a.copy(),
            signal_b=self.arr_b.copy(),
            phase_diff=self.arr_phase.copy(),
            time_axis=self.time_axis.copy(),
            time_seconds=self.core.get_time(),
            meta={
                "effective_freq_a": fa,
                "effective_freq_b": fb,
                "kuramoto_r": self.core.get_sync(),
                "latest_phase_deg": float(np.degrees(self.core.get_phase_diff())),
            },
        )

    def reset(self) -> None:
        self.core.reset(exact_seed_reset=False)

    def apply_params(self, params: Dict[str, float]) -> None:
        self.core.apply_params(params)


class EEGReplaySource(BaseSource):
    mode_name = "EEG Replay (stub)"

    def __init__(self, buf_len: int, dt: float):
        self.buf_len = buf_len
        self.dt = dt
        self.time_axis = np.linspace(-(buf_len - 1) * dt, 0.0, buf_len)
        self._time = 0.0

        self._signal_a = np.zeros(buf_len, dtype=np.float64)
        self._signal_b = np.zeros(buf_len, dtype=np.float64)
        self._phase = np.zeros(buf_len, dtype=np.float64)

        self.loaded = False
        self.file_path: Optional[pathlib.Path] = None

    def load_set_file(self, path: pathlib.Path) -> None:
        """
        Future hook for real EEGLAB loading.

        Planned pipeline:
        - load .set using MNE: mne.io.read_raw_eeglab(..., preload=True)
        - choose channels / ROI aggregation
        - compute slow-alpha / fast-alpha traces per window
        - derive one or more interaction metrics
        - fill internal buffers with time-resolved replay values
        """
        self.file_path = path
        self.loaded = False
        raise NotImplementedError(
            "EEG replay is scaffolded but not implemented yet. "
            "Next step: add MNE-based EEGLAB loading + windowed alpha metric extraction."
        )

    def get_packet(self) -> DataPacket:
        # Placeholder so the app can still switch into EEG mode without crashing.
        self._time += self.dt * STEPS_PER_FRAME
        return DataPacket(
            signal_a=self._signal_a.copy(),
            signal_b=self._signal_b.copy(),
            phase_diff=self._phase.copy(),
            time_axis=self.time_axis.copy(),
            time_seconds=self._time,
            meta={
                "effective_freq_a": 0.0,
                "effective_freq_b": 0.0,
                "kuramoto_r": 0.0,
                "latest_phase_deg": 0.0,
            },
        )

    def reset(self) -> None:
        self._time = 0.0
        self._signal_a.fill(0.0)
        self._signal_b.fill(0.0)
        self._phase.fill(0.0)


# =============================================================================
# Main UI
# =============================================================================

class AlphaApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Alpha Interaction Sandbox")
        self.resize(1250, 840)

        self.core = SimCore(LIB_PATH, BUF_LEN)
        self.metric_registry = MetricRegistry()
        self.param_values = dict(DEFAULT_PARAMS)

        self.demo_source = DemoSource(self.core, BUF_LEN, DT, STEPS_PER_FRAME)
        self.eeg_source = EEGReplaySource(BUF_LEN, DT)
        self.current_source: BaseSource = self.demo_source

        self._paused = False
        self._build_ui()
        self._wire_initial_state()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(TIMER_INTERVAL_MS)

    def closeEvent(self, event):
        try:
            self.core.free()
        finally:
            super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left: plots
        self.graphics = pg.GraphicsLayoutWidget(title="Alpha Interaction Sandbox")
        root.addWidget(self.graphics, stretch=3)

        pg.setConfigOptions(antialias=True)

        self.plot_a = self.graphics.addPlot(row=0, col=0, title="Signal A")
        self.plot_a.setLabel("left", "Amplitude", units="a.u.")
        self.plot_a.showGrid(x=False, y=True, alpha=0.3)
        self.plot_a.setYRange(-4, 4)
        self.curve_a = self.plot_a.plot(pen=pg.mkPen(color=BLUE, width=1.6))

        self.plot_b = self.graphics.addPlot(row=1, col=0, title="Signal B")
        self.plot_b.setLabel("left", "Amplitude", units="a.u.")
        self.plot_b.showGrid(x=False, y=True, alpha=0.3)
        self.plot_b.setYRange(-4, 4)
        self.plot_b.setXLink(self.plot_a)
        self.curve_b = self.plot_b.plot(pen=pg.mkPen(color=GREEN, width=1.6))

        self.plot_phase = self.graphics.addPlot(row=2, col=0, title="Phase difference A − B")
        self.plot_phase.setLabel("left", "Δφ", units="rad")
        self.plot_phase.showGrid(x=False, y=True, alpha=0.3)
        self.plot_phase.setYRange(-np.pi, np.pi)
        self.plot_phase.setXLink(self.plot_a)
        self.plot_phase.addLine(y=0.0, pen=pg.mkPen(GRAY, width=0.8, style=QtCore.Qt.DashLine))
        self.plot_phase.addLine(y=np.pi, pen=pg.mkPen(GRAY, width=0.5, style=QtCore.Qt.DotLine))
        self.plot_phase.addLine(y=-np.pi, pen=pg.mkPen(GRAY, width=0.5, style=QtCore.Qt.DotLine))
        self.curve_phase = self.plot_phase.plot(pen=pg.mkPen(color=PURPLE, width=1.6))

        self.plot_metric = self.graphics.addPlot(row=3, col=0, title="Selected alpha interaction metric")
        self.plot_metric.setLabel("left", "Metric", units="a.u.")
        self.plot_metric.setLabel("bottom", "Time", units="s")
        self.plot_metric.showGrid(x=False, y=True, alpha=0.3)
        self.plot_metric.setXLink(self.plot_a)
        self.curve_metric = self.plot_metric.plot(pen=pg.mkPen(color=ORANGE, width=1.8))

        metric_layout = self.graphics.addLayout(row=4, col=0)
        metric_layout.setContentsMargins(10, 4, 10, 4)
        self.lbl_mode = pg.LabelItem(justify="center")
        self.lbl_fa = pg.LabelItem(justify="center")
        self.lbl_fb = pg.LabelItem(justify="center")
        self.lbl_metric_scalar = pg.LabelItem(justify="center")
        self.lbl_phase = pg.LabelItem(justify="center")
        self.lbl_time = pg.LabelItem(justify="center")
        for col, lbl in enumerate([
            self.lbl_mode,
            self.lbl_fa,
            self.lbl_fb,
            self.lbl_metric_scalar,
            self.lbl_phase,
            self.lbl_time,
        ]):
            metric_layout.addItem(lbl, row=0, col=col)

        # Right: controls
        controls = QtWidgets.QFrame()
        controls.setFrameShape(QtWidgets.QFrame.StyledPanel)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        root.addWidget(controls, stretch=1)

        # Mode selection
        group_mode = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QVBoxLayout(group_mode)
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems([self.demo_source.mode_name, self.eeg_source.mode_name])
        mode_layout.addWidget(self.combo_mode)

        self.btn_load_eeg = QtWidgets.QPushButton("Load EEGLAB .set (future)")
        self.btn_load_eeg.setEnabled(False)
        mode_layout.addWidget(self.btn_load_eeg)
        self.lbl_mode_status = QtWidgets.QLabel("Demo mode active.")
        self.lbl_mode_status.setWordWrap(True)
        mode_layout.addWidget(self.lbl_mode_status)
        controls_layout.addWidget(group_mode)

        # Metric selection
        group_metric = QtWidgets.QGroupBox("Metric")
        metric_controls = QtWidgets.QVBoxLayout(group_metric)
        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItems(self.metric_registry.names)
        metric_controls.addWidget(self.combo_metric)

        self.lbl_metric_hint = QtWidgets.QLabel(
            "This menu is designed so you can swap in whichever alpha interaction "
            "metric survives your actual analyses."
        )
        self.lbl_metric_hint.setWordWrap(True)
        metric_controls.addWidget(self.lbl_metric_hint)
        controls_layout.addWidget(group_metric)

        # Parameters
        group_params = QtWidgets.QGroupBox("Demo parameters")
        params_layout = QtWidgets.QVBoxLayout(group_params)
        self.value_labels: Dict[str, QtWidgets.QLabel] = {}

        for key, label, mn, mx, mult in PARAM_SPECS:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setFixedWidth(135)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(int(mn * mult))
            slider.setMaximum(int(mx * mult))
            slider.setValue(int(self.param_values[key] * mult))
            value_lbl = QtWidgets.QLabel(f"{self.param_values[key]:.2f}")
            value_lbl.setFixedWidth(48)
            slider.valueChanged.connect(self.make_slider_callback(key, mult))
            row.addWidget(lbl)
            row.addWidget(slider)
            row.addWidget(value_lbl)
            params_layout.addLayout(row)
            self.value_labels[key] = value_lbl

        controls_layout.addWidget(group_params)

        # Buttons
        button_row = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        button_row.addWidget(self.btn_reset)
        button_row.addWidget(self.btn_pause)
        controls_layout.addLayout(button_row)

        self.chk_exact_reset = QtWidgets.QCheckBox("Exact seeded reset")
        self.chk_exact_reset.setChecked(False)
        controls_layout.addWidget(self.chk_exact_reset)

        self.lbl_notes = QtWidgets.QLabel(
            "Future EEG replay path:\n"
            "EEGLAB .set → MNE load → ROI/channel selection → slow/fast alpha windowing "
            "→ interaction metric(s) → time-resolved replay"
        )
        self.lbl_notes.setWordWrap(True)
        controls_layout.addWidget(self.lbl_notes)
        controls_layout.addStretch(1)

        # Connections
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.combo_metric.currentIndexChanged.connect(self.on_metric_changed)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        self.btn_load_eeg.clicked.connect(self.on_load_eeg_clicked)

    def _wire_initial_state(self) -> None:
        self.demo_source.apply_params(self.param_values)
        self.refresh_now()

    def make_slider_callback(self, key: str, mult: int):
        def callback(int_value: int) -> None:
            value = int_value / mult
            self.param_values[key] = value
            self.value_labels[key].setText(f"{value:.2f}")
            if self.current_source is self.demo_source:
                self.demo_source.apply_params({key: value})
        return callback

    def on_mode_changed(self) -> None:
        mode = self.combo_mode.currentText()
        if mode == self.demo_source.mode_name:
            self.current_source = self.demo_source
            self.btn_load_eeg.setEnabled(False)
            self.lbl_mode_status.setText("Demo mode active.")
        else:
            self.current_source = self.eeg_source
            self.btn_load_eeg.setEnabled(True)
            self.lbl_mode_status.setText(
                "EEG replay scaffold active. You can keep building the pipeline here "
                "without changing the rest of the UI architecture."
            )
        self.refresh_now()

    def on_metric_changed(self) -> None:
        self.refresh_now()

    def on_pause_clicked(self) -> None:
        self._paused = not self._paused
        self.btn_pause.setText("Resume" if self._paused else "Pause")

    def on_reset_clicked(self) -> None:
        if self.current_source is self.demo_source:
            self.core.reset(exact_seed_reset=self.chk_exact_reset.isChecked())
        else:
            self.current_source.reset()
        self.refresh_now()

    def on_load_eeg_clicked(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select EEGLAB .set file",
            str(HERE),
            "EEGLAB files (*.set);;All files (*)",
        )
        if not path_str:
            return

        try:
            self.eeg_source.load_set_file(pathlib.Path(path_str))
        except NotImplementedError as exc:
            QtWidgets.QMessageBox.information(self, "EEG replay scaffold", str(exc))

    def on_timer(self) -> None:
        if self._paused:
            return
        self.refresh_now()

    def refresh_now(self) -> None:
        packet = self.current_source.get_packet()
        metric_name = self.combo_metric.currentText()
        metric_curve, metric_scalar = self.metric_registry.compute(metric_name, packet)

        self.curve_a.setData(packet.time_axis, packet.signal_a)
        self.curve_b.setData(packet.time_axis, packet.signal_b)
        self.curve_phase.setData(packet.time_axis, packet.phase_diff)
        self.curve_metric.setData(packet.time_axis, metric_curve)

        self.plot_metric.setTitle(f"Selected alpha interaction metric — {metric_name}")

        fa = packet.meta.get("effective_freq_a", 0.0)
        fb = packet.meta.get("effective_freq_b", 0.0)
        phase_deg = packet.meta.get("latest_phase_deg", 0.0)

        self.lbl_mode.setText(f"<b>Mode</b><br>{self.current_source.mode_name}")
        self.lbl_fa.setText(f"<b>Freq A</b><br>{fa:.2f} Hz")
        self.lbl_fb.setText(f"<b>Freq B</b><br>{fb:.2f} Hz")
        self.lbl_metric_scalar.setText(f"<b>{metric_name}</b><br>{metric_scalar:.3f}")
        self.lbl_phase.setText(f"<b>Δφ</b><br>{phase_deg:.1f}°")
        self.lbl_time.setText(f"<b>Time</b><br>{packet.time_seconds:.2f} s")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = AlphaApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())