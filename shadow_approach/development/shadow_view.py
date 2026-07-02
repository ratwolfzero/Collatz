import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

plt.rcParams.update(
    {
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
    }
)
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def collatz_accelerated(n, max_steps=40):
    values = [n]
    parities = []
    x = n
    for _ in range(max_steps):
        if x == 1:
            break
        if x % 2 == 0:
            x = x // 2
            parities.append(0)
        else:
            x = (3 * x + 1) // 2
            parities.append(1)
        values.append(x)
    return values, parities


class CollatzShadowExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collatz Shadow Research Explorer")
        self.resize(1380, 920)

        self.figure = Figure(figsize=(13, 8), dpi=120, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(520)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("QToolBar { spacing: 4px; }")

        self._build_ui()
        self.refresh_plot()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        central.setStyleSheet("background-color: #edf2f7;")
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        controls_group = QGroupBox("Interactive controls")
        controls_group.setStyleSheet("""
QGroupBox {
    font-weight: 600;
    background-color: #f8fbff;
    color: #18324a;
    border: 1px solid #b9c5d4;
    border-radius: 10px;
    margin-top: 8px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #18324a;
}

/* =========================
   FIX INPUT READABILITY
   ========================= */

QLabel {
    color: #18324a;
}

/* Spinboxes (your main issue) */
QSpinBox {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #94a3b8;
    border-radius: 6px;
    padding: 4px 6px;
}

QSpinBox:hover {
    border: 1px solid #64748b;
}

QSpinBox:focus {
    border: 2px solid #2b6cb0;
}

/* ComboBox (preset selector) */
QComboBox {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #94a3b8;
    border-radius: 6px;
    padding: 4px 6px;
}

QComboBox:hover {
    border: 1px solid #64748b;
}

QComboBox:focus {
    border: 2px solid #2b6cb0;
}

/* Dropdown popup readability */
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #111827;
    selection-background-color: #2b6cb0;
    selection-color: white;
    border: 1px solid #94a3b8;
}

/* Optional: makes disabled state readable instead of washed out */
QSpinBox:disabled, QComboBox:disabled {
    background-color: #e5e7eb;
    color: #6b7280;
}
""")
        controls_layout = QVBoxLayout(controls_group)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setVerticalSpacing(8)

        self.start_value_spin = QSpinBox()
        self.start_value_spin.setRange(1, 10**9)
        self.start_value_spin.setValue(27)
        self.start_value_spin.setToolTip("Initial value for the accelerated Collatz orbit")
        form_layout.addRow("Initial value", self.start_value_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 2000)
        self.max_steps_spin.setValue(200)
        form_layout.addRow("Max accelerated steps", self.max_steps_spin)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 200)
        self.window_spin.setValue(10)
        form_layout.addRow("Rolling window", self.window_spin)

        self.block_length_spin = QSpinBox()
        self.block_length_spin.setRange(2, 8)
        self.block_length_spin.setValue(4)
        form_layout.addRow("Block length", self.block_length_spin)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["27", "19", "97", "871", "1003"])
        self.preset_combo.setCurrentText("27")
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        form_layout.addRow("Quick preset", self.preset_combo)

        controls_layout.addLayout(form_layout)

        button_row = QHBoxLayout()
        generate_button = QPushButton("Generate analysis")
        generate_button.setStyleSheet(
            "QPushButton { background-color: #2b6cb0; color: white; border: none; border-radius: 6px; padding: 7px 12px; font-weight: 600; }"
            "QPushButton:hover { background-color: #225a93; }"
        )
        generate_button.clicked.connect(self.refresh_plot)
        button_row.addWidget(generate_button)

        save_button = QPushButton("Save figure")
        save_button.setStyleSheet(
            "QPushButton { background-color: #4a5568; color: white; border: none; border-radius: 6px; padding: 7px 12px; font-weight: 600; }"
            "QPushButton:hover { background-color: #364151; }"
        )
        save_button.clicked.connect(self.save_current_figure)
        button_row.addWidget(save_button)
        button_row.addStretch()
        controls_layout.addLayout(button_row)

        main_layout.addWidget(controls_group)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.summary_label.setMinimumHeight(88)
        self.summary_label.setStyleSheet(
            "background-color: #f4f8fc; border: 1px solid #b9c5d4; border-radius: 10px; padding: 10px; color: #1f3044;"
        )
        main_layout.addWidget(self.summary_label)

        plot_box = QGroupBox("Shadow sequence view")
        plot_box.setStyleSheet(
            "QGroupBox { font-weight: 600; background-color: #f8fbff; color: #18324a; border: 1px solid #b9c5d4; border-radius: 10px; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_box)

    def _apply_preset(self, value):
        try:
            self.start_value_spin.setValue(int(value))
        except ValueError:
            pass

    def refresh_plot(self):
        n = self.start_value_spin.value()
        max_steps = self.max_steps_spin.value()
        window = self.window_spin.value()
        block_length = self.block_length_spin.value()

        values, parities = collatz_accelerated(n, max_steps=max_steps)
        self._draw_dashboard(values, parities, window=window, block_length=block_length)
        self._update_summary(n, values, parities, window=window, block_length=block_length)

    def _draw_dashboard(self, values, parities, window=10, block_length=4):
        self.figure.clear()
        self.figure.set_facecolor("white")
        gs = self.figure.add_gridspec(3, 1, height_ratios=[2.3, 1.0, 1.2], hspace=0.32)

        ax1 = self.figure.add_subplot(gs[0])
        ax1.set_facecolor("#fcfdff")
        ax1.plot(range(len(values)), values, marker="o", markersize=4.5, linewidth=2.2, color="#1f77b4", label="Orbit")
        ax1.set_title("Accelerated orbit and parity shadow")
        ax1.set_xlabel("Accelerated step")
        ax1.set_ylabel("Value")
        ax1.grid(alpha=0.25, linewidth=0.8)
        ax1.tick_params(axis="both", which="major", length=4, width=0.8)

        ax2 = ax1.twinx()
        ax2.step(range(len(parities)), parities, where="mid", color="#d62728", linewidth=1.8, alpha=0.9)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["even", "odd"])
        ax2.set_ylabel("Parity (0 = even, 1 = odd)")
        ax2.grid(False)

        ax3 = self.figure.add_subplot(gs[1])
        ax3.set_facecolor("#fcfdff")
        if len(parities) >= window:
            positions = []
            density = []
            for i in range(len(parities) - window + 1):
                density.append(np.mean(parities[i : i + window]))
                positions.append(i + window / 2)
            ax3.plot(positions, density, linewidth=2.2, color="#ff7f0e")
            ax3.fill_between(positions, density, 0, color="#ff7f0e", alpha=0.16)
        else:
            ax3.text(0.5, 0.5, "Window too large for the available parities", ha="center", va="center", color="#666666")
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("Accelerated step")
        ax3.set_ylabel("Odd-step density")
        ax3.set_title(f"Rolling parity density (window = {window})")
        ax3.grid(alpha=0.25, linewidth=0.8)
        ax3.tick_params(axis="both", which="major", length=4, width=0.8)

        ax4 = self.figure.add_subplot(gs[2])
        ax4.set_facecolor("#fcfdff")
        blocks = ["".join(str(b) for b in parities[i : i + block_length]) for i in range(len(parities) - block_length + 1)]
        counts = Counter(blocks)
        labels = [format(i, f"0{block_length}b") for i in range(2**block_length)]
        frequencies = [counts.get(label, 0) for label in labels]
        ax4.bar(labels, frequencies, color="#2ca02c", alpha=0.9, edgecolor="#1f7f1f", linewidth=0.8)
        ax4.set_xlabel(f"Parity block (length {block_length})")
        ax4.set_ylabel("Occurrences")
        ax4.set_title("Recurrence of parity blocks")
        ax4.grid(axis="y", alpha=0.25, linewidth=0.8)
        for label in ax4.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        ax4.tick_params(axis="both", which="major", length=4, width=0.8)

        self.figure.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.36)
        self.canvas.draw_idle()

    def _update_summary(self, n, values, parities, window=10, block_length=4):
        shadow_word = "".join(str(p) for p in parities)
        odd_count = sum(parities)
        even_count = len(parities) - odd_count
        if len(parities) >= block_length:
            blocks = [shadow_word[i : i + block_length] for i in range(len(shadow_word) - block_length + 1)]
            most_common_block = Counter(blocks).most_common(1)[0] if blocks else ("", 0)
            block_summary = f"Most common block: {most_common_block[0]} ({most_common_block[1]} occurrences)"
        else:
            block_summary = "Most common block: n/a"

        summary = (
            f"Start value: {n}; orbit length: {len(values)}; parity steps: {len(parities)}; "
            f"odd steps: {odd_count}; even steps: {even_count}; window: {window}; "
            f"block length: {block_length}.\n"
            f"Shadow word prefix: {shadow_word[:80]}{'…' if len(shadow_word) > 80 else ''}.\n"
            f"{block_summary}"
        )
        self.summary_label.setText(summary)

    def save_current_figure(self):
        out_dir = os.path.join(os.path.dirname(__file__), "shadow_figures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"collatz_shadow_{self.start_value_spin.value()}_{self.max_steps_spin.value()}.png")
        self.figure.savefig(out_path, dpi=220, bbox_inches="tight")
        QMessageBox.information(self, "Saved", f"Figure saved to {out_path}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = CollatzShadowExplorer()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
