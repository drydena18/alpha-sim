from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

BLUE = (55, 138, 221)
GREEN = (29, 158, 117)
PURPLE = (127, 119, 221)
ORANGE = (220, 143, 63)
GRAY = (160, 160, 160)

class PlotBundle:
    """
    Holds the plot widgets and curves so the main app/controller stays cleaner.
    """

    def __init__(self, graphics_widget: pg.GraphicsLayoutWidget):
        self.graphics = graphics_widget

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

        label_layout = self.graphics.addLayout(row=4, col=0)
        label_layout.setContentsMargins(10, 4, 10, 4)

        self.lbl_mode = pg.LabelItem(justify="center")
        self.lbl_fa = pg.LabelItem(justify="center")
        self.lbl_fb = pg.LabelItem(justify="center")
        self.lbl_metric_scalar = pg.LabelItem(justify="center")
        self.lbl_phase = pg.LabelItem(justify="center")
        self.lbl_time = pg.LabelItem(justify="center")

        labels = [
            self.lbl_mode,
            self.lbl_fa,
            self.lbl_fb,
            self.lbl_metric_scalar,
            self.lbl_phase,
            self.lbl_time,
        ]
        for col, lbl in enumerate(labels):
            label_layout.addItem(lbl, row=0, col=col)

    def update(
        self,
        time_axis,
        signal_a,
        signal_b,
        phase_diff,
        metric_curve,
        metric_name: str,
        mode_name: str,
        freq_a: float,
        freq_b: float,
        metric_scalar: float,
        metric_units: str,
        phase_deg: float,
        time_seconds: float,
    ) -> None:
        self.curve_a.setData(time_axis, signal_a)
        self.curve_b.setData(time_axis, signal_b)
        self.curve_phase.setData(time_axis, phase_diff)
        self.curve_metric.setData(time_axis, metric_curve)

        self.plot_metric.setText(f"Selected alpha interaction metric: {metric_name}")

        self.lbl_mode.setText(f"<b>Mode</b><br>{mode_name}")
        self.lbl_fa.setText(f"<b>Freq A</b><br>{freq_a:.2f} Hz")
        self.lbl_fb.setText(f"<b>Freq B</b><br>{freq_b:.2f} Hz")
        self.lbl_metric_scalar.setText(f"<b>{metric_name}</b><br>{metric_scalar:.3f} {metric_units}")
        self.lbl_phase.setText(f"<b>Δφ</b><br>{phase_deg:.1f}°")
        self.lbl_time.setText(f"<b>Time</b><br>{time_seconds:.2f} s")