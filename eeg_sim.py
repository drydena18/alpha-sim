"""
eeg_sim.py

Main entrypoint for the alpha interaction sandbox.

Current Architecture:
---------------------
- app/              -> GUI-facing helpers, state models, plotting, controller
- this file         -> ctypes bridge, demo source, EEG replay stub, main window
- vdp_core.so / .c  -> native van der Pol simulation backend

Notes
-----
For now, SimCore + DemoSource + EEGReplaySource remain here to be refactored 
incrementally without breaking the working app. Later, the real EEG logic will
move into eeg/ modules.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
from typing import Dict, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from app.app_controller import AppController
from app.metric_registry import build_default_metric_registry
from app.plotting import PlotBundle
from app.state_models import AppConfig, DataPacket
from app.ui_controls import populate_param_group


# =============================================================================
# Config
# =============================================================================

HERE = pathlib.Path(__file__).resolve().parent

CONFIG = AppConfig(
    buf_len=1000,
    dt=1.0 / 1000.0,
    steps_per_frame=6,
    timer_interval_ms=16,
    lib_path=str(HERE / "vdp_core.so"),
)

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
    def __init__(self, lib_path: str, buf_len: int):
        self.lib_path = pathlib.Path(lib_path)
        if not self.lib_path.exists():
            sys.exit(
                "vdp_core.so not found. Build it first, e.g.:\n"
                "  gcc -O2 -march=native -shared -fPIC -o vdp_core.so vdp_core.c -lm"
            )

        self.lib = ctypes.CDLL(str(self.lib_path))
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

        # Optional newer API; fall back gracefully if absent
        self.has_seed_reset = hasattr(lib, "sim_reset_with_seed")
        if self.has_seed_reset:
            lib.sim_reset_with_seed.restype = None
            lib.sim_reset_with_seed.argtypes = [ctypes.c_void_p]

        lib.sim_advance.restype = None
        lib.sim_advance.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]

        lib.sim_get_buffers.restype = None

        # Support both old and new signatures
        try:
            lib.sim_get_buffers.argtypes = [
                ctypes.c_void_p,
                _c_double_p,
                _c_double_p,
                _c_double_p,
                ctypes.c_int,
            ]
            self.buffer_api_has_len = True
        except Exception:
            lib.sim_get_buffers.argtypes = [
                ctypes.c_void_p,
                _c_double_p,
                _c_double_p,
                _c_double_p,
            ]
            self.buffer_api_has_len = False

        for getter in [
            "sim_sync_index",
            "sim_get_time",
        ]:
            getattr(lib, getter).restype = ctypes.c_double
            getattr(lib, getter).argtypes = [ctypes.c_void_p]

        self.has_phase_getter = hasattr(lib, "sim_get_phase_diff")
        if self.has_phase_getter:
            lib.sim_get_phase_diff.restype = ctypes.c_double
            lib.sim_get_phase_diff.argtypes = [ctypes.c_void_p]

        self.has_eff_freq_getters = (
            hasattr(lib, "sim_get_effective_freq_a")
            and hasattr(lib, "sim_get_effective_freq_b")
        )
        if self.has_eff_freq_getters:
            lib.sim_get_effective_freq_a.restype = ctypes.c_double
            lib.sim_get_effective_freq_a.argtypes = [ctypes.c_void_p]
            lib.sim_get_effective_freq_b.restype = ctypes.c_double
            lib.sim_get_effective_freq_b.argtypes = [ctypes.c_void_p]

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
        if exact_seed_reset and self.has_seed_reset:
            self.lib.sim_reset_with_seed(self.ctx)
        else:
            self.lib.sim_reset(self.ctx)

    def advance(self, steps: int, dt: float) -> None:
        self.lib.sim_advance(self.ctx, steps, dt)

    def get_buffers(self, out_a: np.ndarray, out_b: np.ndarray, out_phase: np.ndarray) -> None:
        if self.buffer_api_has_len:
            self.lib.sim_get_buffers(
                self.ctx,
                out_a.ctypes.data_as(_c_double_p),
                out_b.ctypes.data_as(_c_double_p),
                out_phase.ctypes.data_as(_c_double_p),
                self.buf_len,
            )
        else:
            self.lib.sim_get_buffers(
                self.ctx,
                out_a.ctypes.data_as(_c_double_p),
                out_b.ctypes.data_as(_c_double_p),
                out_phase.ctypes.data_as(_c_double_p),
            )

    def get_sync(self) -> float:
        return float(self.lib.sim_sync_index(self.ctx))

    def get_time(self) -> float:
        return float(self.lib.sim_get_time(self.ctx))

    def get_phase_diff(self) -> float:
        if self.has_phase_getter:
            return float(self.lib.sim_get_phase_diff(self.ctx))
        return 0.0

    def get_effective_freqs(self, fallback_params: Dict[str, float]) -> tuple[float, float]:
        if self.has_eff_freq_getters:
            fa = float(self.lib.sim_get_effective_freq_a(self.ctx))
            fb = float(self.lib.sim_get_effective_freq_b(self.ctx))
            return fa, fb

        fa = fallback_params["freq_a"] + fallback_params["alertness_a"] * 1.5
        fb = fallback_params["freq_b"] + fallback_params["alertness_b"] * 1.5
        return fa, fb


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

    def __init__(self, core: SimCore, cfg: AppConfig, params_ref: Dict[str, float]):
        self.core = core
        self.cfg = cfg
        self.params_ref = params_ref

        self.arr_a = np.zeros(cfg.buf_len, dtype=np.float64)
        self.arr_b = np.zeros(cfg.buf_len, dtype=np.float64)
        self.arr_phase = np.zeros(cfg.buf_len, dtype=np.float64)
        self.time_axis = np.linspace(-(cfg.buf_len - 1) * cfg.dt, 0.0, cfg.buf_len)

    def get_packet(self) -> DataPacket:
        self.core.advance(self.cfg.steps_per_frame, self.cfg.dt)
        self.core.get_buffers(self.arr_a, self.arr_b, self.arr_phase)
        fa, fb = self.core.get_effective_freqs(self.params_ref)

        latest_phase_deg = float(np.degrees(self.arr_phase[-1])) if len(self.arr_phase) else 0.0
        if self.core.has_phase_getter:
            latest_phase_deg = float(np.degrees(self.core.get_phase_diff()))

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
                "latest_phase_deg": latest_phase_deg,
            },
        )

    def reset(self) -> None:
        self.core.reset(exact_seed_reset=False)

    def reset_exact(self) -> None:
        self.core.reset(exact_seed_reset=True)

    def apply_params(self, params: Dict[str, float]) -> None:
        self.core.apply_params(params)


class EEGReplaySource(BaseSource):
    mode_name = "EEG Replay (stub)"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.time_axis = np.linspace(-(cfg.buf_len - 1) * cfg.dt, 0.0, cfg.buf_len)
        self._time = 0.0

        self._signal_a = np.zeros(cfg.buf_len, dtype=np.float64)
        self._signal_b = np.zeros(cfg.buf_len, dtype=np.float64)
        self._phase = np.zeros(cfg.buf_len, dtype=np.float64)

        self.loaded = False
        self.file_path: Optional[pathlib.Path] = None

    def load_set_file(self, path: pathlib.Path) -> None:
        self.file_path = path
        self.loaded = False
        raise NotImplementedError(
            "EEG replay is scaffolded but not implemented yet.\n"
            "Next step: eeg_loader.py + preprocess_pipeline.py + replay_pipeline.py"
        )

    def get_packet(self) -> DataPacket:
        self._time += self.cfg.dt * self.cfg.steps_per_frame
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
# Main window
# =============================================================================

class AlphaApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Alpha Interaction Sandbox")
        self.resize(1280, 860)

        pg.setConfigOptions(antialias=True)

        self.param_values = dict(DEFAULT_PARAMS)
        self.metric_registry = build_default_metric_registry()
        self.controller = AppController(self.metric_registry)

        self.core = SimCore(CONFIG.lib_path, CONFIG.buf_len)
        self.demo_source = DemoSource(self.core, CONFIG, self.param_values)
        self.eeg_source = EEGReplaySource(CONFIG)
        self.current_source = self.demo_source

        self.controller.set_source(self.current_source)
        self.controller.set_metric(self.metric_registry.names()[0])

        self._paused = False
        self._build_ui()
        self.controller.apply_params(self.param_values)
        self.refresh_now()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(CONFIG.timer_interval_ms)

    def closeEvent(self, event):
        try:
            self.timer.stop()
            self.core.free()
        finally:
            super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left side: plots
        self.graphics = pg.GraphicsLayoutWidget(title="Alpha Interaction Sandbox")
        root.addWidget(self.graphics, stretch=3)
        self.plots = PlotBundle(self.graphics)

        # Right side: controls
        controls = QtWidgets.QFrame()
        controls.setFrameShape(QtWidgets.QFrame.StyledPanel)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        root.addWidget(controls, stretch=1)

        # Mode
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

        # Metric
        group_metric = QtWidgets.QGroupBox("Metric")
        metric_layout = QtWidgets.QVBoxLayout(group_metric)

        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItems(self.metric_registry.names())
        metric_layout.addWidget(self.combo_metric)

        self.lbl_metric_hint = QtWidgets.QLabel(
            "Metric menu is scaffolded so you can later swap in whichever "
            "alpha interaction metric survives your thesis."
        )
        self.lbl_metric_hint.setWordWrap(True)
        metric_layout.addWidget(self.lbl_metric_hint)

        controls_layout.addWidget(group_metric)

        # Params
        self.group_params = QtWidgets.QGroupBox("Demo parameters")
        _, self.param_sliders, self.value_labels = populate_param_group(
            self.group_params,
            PARAM_SPECS,
            self.param_values,
            self.make_slider_callback,
        )
        controls_layout.addWidget(self.group_params)

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
            ".set load → preprocessing → channel/ROI selection → spectral extraction "
            "→ alpha metric(s) → time-resolved replay"
        )
        self.lbl_notes.setWordWrap(True)
        controls_layout.addWidget(self.lbl_notes)
        controls_layout.addStretch(1)

        # Wiring
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.combo_metric.currentTextChanged.connect(self.on_metric_changed)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        self.btn_load_eeg.clicked.connect(self.on_load_eeg_clicked)

    def make_slider_callback(self, key: str, mult: int):
        def callback(int_value: int) -> None:
            value = int_value / mult
            self.param_values[key] = value
            self.value_labels[key].setText(f"{value:.2f}")

            if self.current_source is self.demo_source:
                self.controller.apply_params({key: value})

        return callback

    def on_mode_changed(self) -> None:
        mode = self.combo_mode.currentText()

        if mode == self.demo_source.mode_name:
            self.current_source = self.demo_source
            self.btn_load_eeg.setEnabled(False)
            self.group_params.setEnabled(True)
            self.lbl_mode_status.setText("Demo mode active.")
        else:
            self.current_source = self.eeg_source
            self.btn_load_eeg.setEnabled(True)
            self.group_params.setEnabled(False)
            self.lbl_mode_status.setText(
                "EEG replay scaffold active. The GUI is ready for a future replay pipeline."
            )

        self.controller.set_source(self.current_source)
        self.refresh_now()

    def on_metric_changed(self, metric_name: str) -> None:
        self.controller.set_metric(metric_name)
        self.refresh_now()

    def on_pause_clicked(self) -> None:
        self._paused = not self._paused
        self.btn_pause.setText("Resume" if self._paused else "Pause")

    def on_reset_clicked(self) -> None:
        if self.current_source is self.demo_source:
            if self.chk_exact_reset.isChecked():
                self.demo_source.reset_exact()
            else:
                self.demo_source.reset()
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
        packet = self.controller.get_current_packet()
        metric_result = self.controller.compute_metric(packet)

        freq_a = packet.meta.get("effective_freq_a", 0.0)
        freq_b = packet.meta.get("effective_freq_b", 0.0)
        phase_deg = packet.meta.get("latest_phase_deg", 0.0)

        self.plots.update(
            time_axis=packet.time_axis,
            signal_a=packet.signal_a,
            signal_b=packet.signal_b,
            phase_diff=packet.phase_diff,
            metric_curve=metric_result.curve,
            metric_name=metric_result.name,
            mode_name=self.current_source.mode_name,
            freq_a=freq_a,
            freq_b=freq_b,
            metric_scalar=metric_result.scalar,
            metric_units=metric_result.units,
            phase_deg=phase_deg,
            time_seconds=packet.time_seconds,
        )


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = AlphaApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())