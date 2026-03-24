from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from pyqttgraph.Qt import QtCore, QtWidgets

ParamSpec = Tuple[str, str, float, float, int]


def build_labeled_slider_row(
    label_text: str,
    min_val: float,
    max_val: float,
    multiplier: int,
    initial_value: float,
    on_change: Callable[[int], None],
):
    row = QtWidgets.QHBoxLayout()

    label = QtWidgets.QLabel(label_text)
    label.setFixedWidth(135)

    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(int(min_val * multiplier))
    slider.setMaximum(int(max_val * multiplier))
    slider.setValue(int(initial_value * multiplier))
    slider.valueChanged.connect(on_change)

    value_label = QtWidgets.QLabel(f"{initial_value:.2f}")
    value_label.setFixedWidth(48)

    row.addWidget(label)
    row.addWidget(slider)
    row.addWidget(value_label)

    return row, slider, value_label


def populate_param_group(
    group_box: QtWidgets.QGroupBox,
    param_specs: List[ParamSpec],
    param_values: Dict[str, float],
    callback_factory: Callable[[str, int], Callable[[int], None]],
):
    layout = QtWidgets.QVBoxLayout(group_box)
    sliders = {}
    value_labels = {}

    for key, label, mn, mx, mult in param_specs:
        row, slider, value_label = build_labeled_slider_row(
            label_text=label,
            min_val=mn,
            max_val=mx,
            multiplier=mult,
            initial_value=param_values[key],
            on_change=callback_factory(key, mult),
        )
        layout.addLayout(row)
        sliders[key] = slider
        value_labels[key] = value_label

    return layout, sliders, value_labels