"""
eeg_sim.py
Real-time EEG alpha-wave simulator.

Architecture
------------
  vdp_core.so  (C)  — RK4 van der Pol engine, ring buffers, sync index
  This file     (Python) — ctypes bridge + PyQtGraph GUI

Dependencies
------------
  pip install pyqtgraph PyQt5 numpy

Build the C library first:
  gcc -O2 -march=native -shared -fPIC -o vdp_core.so vdp_core.c -lm

Then run:
  python eeg_sim.py
"""

import ctypes, os, sys, pathlib
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# ── 1. Load the shared library ───────────────────────────────────────────

_HERE = pathlib.Path(__file__).parent
_LIB_PATH = _HERE / "vdp_core.so"

if not _LIB_PATH.exists():
    sys.exit(
        "vdp_core.so not found.  Build it first:\n"
        "  gcc -O2 -march=native -shared -fPIC -o vdp_core.so vdp_core.c -lm"
    )

lib = ctypes.CDLL(str(_LIB_PATH))

# ── 2. ctypes signatures ─────────────────────────────────────────────────

_c_double_p = ctypes.POINTER(ctypes.c_double)

lib.sim_create.restype  = ctypes.c_void_p
lib.sim_create.argtypes = [ctypes.c_int]

lib.sim_free.restype    = None
lib.sim_free.argtypes   = [ctypes.c_void_p]

lib.sim_reset.restype   = None
lib.sim_reset.argtypes  = [ctypes.c_void_p]

lib.sim_advance.restype  = None
lib.sim_advance.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]

lib.sim_get_buffers.restype  = None
lib.sim_get_buffers.argtypes = [ctypes.c_void_p,
                                 _c_double_p, _c_double_p, _c_double_p]

lib.sim_sync_index.restype  = ctypes.c_double
lib.sim_sync_index.argtypes = [ctypes.c_void_p]

lib.sim_get_time.restype  = ctypes.c_double
lib.sim_get_time.argtypes = [ctypes.c_void_p]

for name, typ in [
    ("sim_set_freq_a",      ctypes.c_double),
    ("sim_set_freq_b",      ctypes.c_double),
    ("sim_set_mu_a",        ctypes.c_double),
    ("sim_set_mu_b",        ctypes.c_double),
    ("sim_set_amp_a",       ctypes.c_double),
    ("sim_set_amp_b",       ctypes.c_double),
    ("sim_set_coupling",    ctypes.c_double),
    ("sim_set_noise",       ctypes.c_double),
    ("sim_set_alertness_a", ctypes.c_double),
    ("sim_set_alertness_b", ctypes.c_double),
]:
    getattr(lib, name).restype  = None
    getattr(lib, name).argtypes = [ctypes.c_void_p, typ]

# ── 3. Simulation context ────────────────────────────────────────────────

BUF_LEN  = 1000          # samples kept in ring buffer
DT       = 1.0 / 1000.0  # integration step (1 ms)
STEPS_PER_FRAME = 6       # C steps per Qt timer tick (~60 Hz → 360 Hz sim)

ctx = lib.sim_create(BUF_LEN)

# Pre-allocate output arrays once (avoids per-frame allocation)
_arr_a   = np.zeros(BUF_LEN, dtype=np.float64)
_arr_b   = np.zeros(BUF_LEN, dtype=np.float64)
_arr_ph  = np.zeros(BUF_LEN, dtype=np.float64)

_ptr_a  = _arr_a.ctypes.data_as(_c_double_p)
_ptr_b  = _arr_b.ctypes.data_as(_c_double_p)
_ptr_ph = _arr_ph.ctypes.data_as(_c_double_p)

x_axis = np.linspace(0, BUF_LEN * DT * STEPS_PER_FRAME, BUF_LEN)

# ── 4. GUI ───────────────────────────────────────────────────────────────

pg.setConfigOptions(antialias=True)
app = QtWidgets.QApplication(sys.argv)

win = pg.GraphicsLayoutWidget(title="EEG Alpha-Wave Simulator")
win.resize(1000, 700)
win.setWindowTitle("EEG Alpha-Wave Simulator  —  van der Pol × 2")

# Colour scheme
BLUE   = (55, 138, 221)
GREEN  = (29, 158, 117)
PURPLE = (127, 119, 221)
GRAY   = (160, 160, 160)

# ── waveform plots ──
plot_a = win.addPlot(row=0, col=0, title="Oscillator A")
plot_a.setYRange(-4, 4); plot_a.setLabel('left', 'Amplitude', units='a.u.')
plot_a.showGrid(x=False, y=True, alpha=0.3)
curve_a = plot_a.plot(pen=pg.mkPen(color=BLUE, width=1.5))

plot_b = win.addPlot(row=1, col=0, title="Oscillator B")
plot_b.setYRange(-4, 4); plot_b.setLabel('left', 'Amplitude', units='a.u.')
plot_b.showGrid(x=False, y=True, alpha=0.3)
curve_b = plot_b.plot(pen=pg.mkPen(color=GREEN, width=1.5))
plot_b.setXLink(plot_a)

plot_ph = win.addPlot(row=2, col=0, title="Phase difference A − B")
plot_ph.setYRange(-np.pi, np.pi)
plot_ph.addLine(y=0,    pen=pg.mkPen(GRAY, width=0.8, style=QtCore.Qt.DashLine))
plot_ph.addLine(y=np.pi, pen=pg.mkPen(GRAY, width=0.5, style=QtCore.Qt.DotLine))
plot_ph.addLine(y=-np.pi,pen=pg.mkPen(GRAY, width=0.5, style=QtCore.Qt.DotLine))
plot_ph.setLabel('left', 'Δφ', units='rad')
curve_ph = plot_ph.plot(pen=pg.mkPen(color=PURPLE, width=1.5))
plot_ph.setXLink(plot_a)

# ── metrics labels ──
metrics_layout = win.addLayout(row=3, col=0)
metrics_layout.setContentsMargins(10, 4, 10, 4)

lbl_fa    = pg.LabelItem(justify='center')
lbl_fb    = pg.LabelItem(justify='center')
lbl_sync  = pg.LabelItem(justify='center')
lbl_dph   = pg.LabelItem(justify='center')
lbl_time  = pg.LabelItem(justify='center')

for col, lbl in enumerate([lbl_fa, lbl_fb, lbl_sync, lbl_dph, lbl_time]):
    metrics_layout.addItem(lbl, row=0, col=col)

def update_metrics():
    sync = lib.sim_sync_index(ctx)
    t    = lib.sim_get_time(ctx)
    fa   = float(lib.sim_get_time.argtypes[0].__class__.__name__)  # placeholder
    dph  = float(_arr_ph[-1])

    # Effective frequencies (natural + alertness boost)
    freq_a_eff = _slider_vals["freq_a"] + _slider_vals["alertness_a"] * 1.5
    freq_b_eff = _slider_vals["freq_b"] + _slider_vals["alertness_b"] * 1.5

    lbl_fa  .setText(f"<b>Freq A</b><br>{freq_a_eff:.2f} Hz")
    lbl_fb  .setText(f"<b>Freq B</b><br>{freq_b_eff:.2f} Hz")
    lbl_sync.setText(f"<b>Sync R</b><br>{sync:.3f}")
    lbl_dph .setText(f"<b>Δφ</b><br>{np.degrees(dph):.1f}°")
    lbl_time.setText(f"<b>Time</b><br>{t:.1f} s")

# ── 5. Control panel (separate window) ──────────────────────────────────

ctrl_win = QtWidgets.QWidget()
ctrl_win.setWindowTitle("Parameters")
ctrl_win.resize(340, 560)

layout = QtWidgets.QVBoxLayout(ctrl_win)

# Slider registry  {key: current_value}
_slider_vals = {
    "freq_a":      10.0,
    "freq_b":       9.0,
    "mu_a":         0.30,
    "mu_b":         0.30,
    "amp_a":        1.00,
    "amp_b":        1.00,
    "coupling":     0.05,
    "noise":        0.05,
    "alertness_a":  0.50,
    "alertness_b":  0.50,
}

_setter_map = {
    "freq_a":      lib.sim_set_freq_a,
    "freq_b":      lib.sim_set_freq_b,
    "mu_a":        lib.sim_set_mu_a,
    "mu_b":        lib.sim_set_mu_b,
    "amp_a":       lib.sim_set_amp_a,
    "amp_b":       lib.sim_set_amp_b,
    "coupling":    lib.sim_set_coupling,
    "noise":       lib.sim_set_noise,
    "alertness_a": lib.sim_set_alertness_a,
    "alertness_b": lib.sim_set_alertness_b,
}

_param_specs = [
    # (key,            label,              min,  max,   step, multiplier)
    ("freq_a",       "Freq A (Hz)",        7.0, 13.0, 0.1,  10),
    ("mu_a",         "Damping A (μ)",      0.05, 2.0, 0.05, 100),
    ("amp_a",        "Amplitude A",        0.2,  3.0, 0.1,  10),
    ("alertness_a",  "Alertness A",        0.0,  1.0, 0.05, 100),
    ("freq_b",       "Freq B (Hz)",        7.0, 13.0, 0.1,  10),
    ("mu_b",         "Damping B (μ)",      0.05, 2.0, 0.05, 100),
    ("amp_b",        "Amplitude B",        0.2,  3.0, 0.1,  10),
    ("alertness_b",  "Alertness B",        0.0,  1.0, 0.05, 100),
    ("coupling",     "Coupling K",         0.0,  0.5, 0.01, 100),
    ("noise",        "Noise σ",            0.0,  0.4, 0.01, 100),
]

_value_labels = {}

def make_slider_callback(key, mult, setter):
    def cb(int_val):
        val = int_val / mult
        _slider_vals[key] = val
        setter(ctx, val)
        _value_labels[key].setText(f"{val:.2f}")
    return cb

for key, label, mn, mx, step, mult in _param_specs:
    row = QtWidgets.QHBoxLayout()
    lbl = QtWidgets.QLabel(label)
    lbl.setFixedWidth(140)
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(int(mn * mult))
    slider.setMaximum(int(mx * mult))
    slider.setValue(int(_slider_vals[key] * mult))
    val_lbl = QtWidgets.QLabel(f"{_slider_vals[key]:.2f}")
    val_lbl.setFixedWidth(40)
    _value_labels[key] = val_lbl
    slider.valueChanged.connect(make_slider_callback(key, mult, _setter_map[key]))
    row.addWidget(lbl); row.addWidget(slider); row.addWidget(val_lbl)
    layout.addLayout(row)

# Buttons
btn_row = QtWidgets.QHBoxLayout()
btn_reset = QtWidgets.QPushButton("Reset")
btn_pause = QtWidgets.QPushButton("Pause")
btn_row.addWidget(btn_reset); btn_row.addWidget(btn_pause)
layout.addLayout(btn_row)

_paused = [False]

def on_reset():
    lib.sim_reset(ctx)

def on_pause():
    _paused[0] = not _paused[0]
    btn_pause.setText("Resume" if _paused[0] else "Pause")

btn_reset.clicked.connect(on_reset)
btn_pause.clicked.connect(on_pause)

ctrl_win.show()

# ── 6. Timer-driven update loop ──────────────────────────────────────────

def update():
    if _paused[0]:
        return

    lib.sim_advance(ctx, STEPS_PER_FRAME, DT)
    lib.sim_get_buffers(ctx, _ptr_a, _ptr_b, _ptr_ph)

    curve_a.setData(x_axis, _arr_a)
    curve_b.setData(x_axis, _arr_b)
    curve_ph.setData(x_axis, _arr_ph)
    update_metrics()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)   # ~60 fps

# ── 7. Cleanup ───────────────────────────────────────────────────────────

win.show()

def cleanup():
    timer.stop()
    lib.sim_free(ctx)

app.aboutToQuit.connect(cleanup)

if __name__ == "__main__":
    sys.exit(app.exec_())