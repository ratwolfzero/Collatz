import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
        self.resize(1280, 840)

        self.figure = Figure(figsize=(11, 7), dpi=120)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._build_ui()
        self.refresh_plot()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        plot_container = QWidget(self)
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("background: #f8f9fa; padding: 8px; border-radius: 4px;")
        plot_layout.addWidget(self.summary_label)

        plot_box = QGroupBox("Shadow sequence view")
        plot_box_layout = QVBoxLayout(plot_box)
        plot_box_layout.setContentsMargins(8, 8, 8, 8)
        plot_box_layout.setSpacing(4)
        plot_box_layout.addWidget(self.toolbar, alignment=Qt.AlignmentFlag.AlignLeft)
        plot_box_layout.addWidget(self.canvas)
        plot_layout.addWidget(plot_box)

        controls_scroll = QScrollArea(self)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(320)
        controls_scroll.setMaximumWidth(360)

        controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        controls_group = QGroupBox("Interactive controls")
        controls_group_layout = QVBoxLayout(controls_group)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

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

        controls_group_layout.addLayout(form_layout)

        panel_group = QGroupBox("Visible panels")
        panel_layout = QVBoxLayout(panel_group)
        self.panel_checkboxes = {
            "orbit": QCheckBox("Orbit + shadow", checked=True),
            "density": QCheckBox("Rolling density", checked=True),
            "blocks": QCheckBox("Block recurrence", checked=True),
            "transition": QCheckBox("Transition heatmap", checked=True),
        }
        for checkbox in self.panel_checkboxes.values():
            panel_layout.addWidget(checkbox)
        controls_group_layout.addWidget(panel_group)

        button_row = QHBoxLayout()
        generate_button = QPushButton("Generate analysis")
        generate_button.clicked.connect(self.refresh_plot)
        button_row.addWidget(generate_button)

        save_button = QPushButton("Save figure")
        save_button.clicked.connect(self.save_current_figure)
        button_row.addWidget(save_button)
        button_row.addStretch()
        controls_group_layout.addLayout(button_row)
        controls_layout.addWidget(controls_group)
        controls_scroll.setWidget(controls_widget)

        main_layout.addWidget(plot_container, stretch=1)
        main_layout.addWidget(controls_scroll, stretch=0)

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

    def _selected_panels(self):
        return [name for name, checkbox in self.panel_checkboxes.items() if checkbox.isChecked()]

    def _draw_transition_heatmap(self, ax, parities, block_length=4):
        if len(parities) < block_length + 1:
            ax.text(0.5, 0.5, "Not enough parity steps for transitions", ha="center", va="center")
            ax.set_title(f"Parity block transitions (length {block_length})")
            return

        blocks = ["".join(str(b) for b in parities[i : i + block_length]) for i in range(len(parities) - block_length + 1)]
        transitions = Counter(zip(blocks[:-1], blocks[1:]))
        labels = [format(i, f"0{block_length}b") for i in range(2**block_length)]
        index = {label: i for i, label in enumerate(labels)}

        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for (src, dst), count in transitions.items():
            if src in index and dst in index:
                matrix[index[src], index[dst]] = count

        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(f"Parity block transitions (length {block_length})")
        ax.set_xlabel("Next block")
        ax.set_ylabel("Current block")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.figure.colorbar(image, ax=ax, shrink=0.92, pad=0.03)

    def _draw_dashboard(self, values, parities, window=10, block_length=4):
        self.figure.clear()
        selected_panels = self._selected_panels()

        if not selected_panels:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one panel to display", ha="center", va="center")
            ax.set_axis_off()
            self.figure.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.06)
            self.canvas.draw_idle()
            return

        gs = self.figure.add_gridspec(len(selected_panels), 1, hspace=0.34)
        panel_heights = {"orbit": 2.2, "density": 1.0, "blocks": 1.0, "transition": 1.2}
        heights = [panel_heights.get(panel, 1.0) for panel in selected_panels]
        gs = self.figure.add_gridspec(len(selected_panels), 1, height_ratios=heights, hspace=0.34)

        for index, panel in enumerate(selected_panels):
            ax = self.figure.add_subplot(gs[index])
            if panel == "orbit":
                ax.plot(range(len(values)), values, marker="o", linewidth=2.0, color="#1f77b4", label="Orbit")
                ax.set_title("Accelerated orbit and parity shadow")
                ax.set_xlabel("Accelerated step")
                ax.set_ylabel("Value")
                ax.grid(alpha=0.3)

                ax2 = ax.twinx()
                ax2.step(range(len(parities)), parities, where="mid", color="#d62728", linewidth=1.6, alpha=0.85)
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(["even", "odd"])
                ax2.set_ylabel("Parity (0 = even, 1 = odd)")
            elif panel == "density":
                if len(parities) >= window:
                    positions = []
                    density = []
                    for i in range(len(parities) - window + 1):
                        density.append(np.mean(parities[i : i + window]))
                        positions.append(i + window / 2)
                    ax.plot(positions, density, linewidth=2.0, color="#ff7f0e")
                else:
                    ax.text(0.5, 0.5, "Window too large for the available parities", ha="center", va="center")
                ax.set_ylim(0, 1)
                ax.set_xlabel("Accelerated step")
                ax.set_ylabel("Odd-step density")
                ax.set_title(f"Rolling parity density (window = {window})")
                ax.grid(alpha=0.3)
            elif panel == "blocks":
                blocks = ["".join(str(b) for b in parities[i : i + block_length]) for i in range(len(parities) - block_length + 1)]
                counts = Counter(blocks)
                labels = [format(i, f"0{block_length}b") for i in range(2**block_length)]
                frequencies = [counts.get(label, 0) for label in labels]
                ax.bar(labels, frequencies, color="#2ca02c", alpha=0.9)
                ax.set_xlabel(f"Parity block (length {block_length})")
                ax.set_ylabel("Occurrences")
                ax.set_title("Recurrence of parity blocks")
                ax.grid(axis="y", alpha=0.3)
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")
            elif panel == "transition":
                self._draw_transition_heatmap(ax, parities, block_length=block_length)

        self.figure.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.06, hspace=0.34)
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
