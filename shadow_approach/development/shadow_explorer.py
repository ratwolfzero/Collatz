import os
import sys
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFontDatabase, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("QtAgg")

plt.rcParams.update(
    {
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
    }
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

        # Core Matplotlib Setup
        self.figure = Figure(dpi=120, facecolor="#1e1e1e", constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._build_ui()
        self.refresh_plot()

    def _build_ui(self):
        # 1. Main Container & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 2. Use a QSplitter to cleanly separate Workspace and Controls
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 3. Build Panels
        plot_panel = self._create_plot_panel()
        control_panel = self._create_control_panel()

        splitter.addWidget(plot_panel)
        splitter.addWidget(control_panel)

        # Set standard initial proportional sizes (75% plot workspace, 25% controls)
        splitter.setSizes([1000, 330])

    def _create_plot_panel(self):
        """Constructs the primary workspace panel containing stats and visualization."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Summary Metrics Header
        self.summary_label = QLabel("Initializing analysis...")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.summary_label.setMinimumHeight(70)
        # Gentle frame styling that respects global palette borders
        self.summary_label.setStyleSheet("QLabel { background-color: #1e1e1e; border: 1px solid #333333; border-radius: 6px; padding: 10px; font-family: monospace; }")
        layout.addWidget(self.summary_label)

        # Visualization View Frame
        plot_box = QGroupBox("Shadow Sequence Workspace")
        plot_box_layout = QVBoxLayout(plot_box)
        plot_box_layout.setContentsMargins(8, 8, 8, 8)
        
        plot_box_layout.addWidget(self.toolbar)
        plot_box_layout.addWidget(self.canvas)
        
        layout.addWidget(plot_box, stretch=1)
        return panel

    def _create_control_panel(self):
        """Constructs the sidebar containing configuration controls inside a unified layout."""
        # Scroll Area for high adaptability across display scales
        scroll_area = QScrollArea()
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(14)

        # Parameters Configuration Group
        param_group = QGroupBox("Sequence Parameters")
        form_layout = QFormLayout(param_group)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setSpacing(8)

        self.start_value_spin = QSpinBox()
        self.start_value_spin.setRange(1, 10**9)
        self.start_value_spin.setValue(27)
        form_layout.addRow("Initial Value:", self.start_value_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 2000)
        self.max_steps_spin.setValue(200)
        form_layout.addRow("Max Steps:", self.max_steps_spin)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 200)
        self.window_spin.setValue(10)
        form_layout.addRow("Rolling Window:", self.window_spin)

        self.block_length_spin = QSpinBox()
        self.block_length_spin.setRange(2, 8)
        self.block_length_spin.setValue(4)
        form_layout.addRow("Block Length:", self.block_length_spin)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["27", "19", "97", "871", "1003"])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        form_layout.addRow("Quick Preset:", self.preset_combo)
        layout.addWidget(param_group)

        # Panels Toggle Visibility Group
        visibility_group = QGroupBox("Display Subplots")
        visibility_layout = QVBoxLayout(visibility_group)
        visibility_layout.setSpacing(6)
        
        self.panel_checkboxes = {
            "orbit": QCheckBox("Orbit + Shadow Track", checked=True),
            "density": QCheckBox("Rolling Density Analysis", checked=True),
            "blocks": QCheckBox("Block Recurrence Metric", checked=True),
            "transition": QCheckBox("Transition Probability Heatmap", checked=True),
        }
        for checkbox in self.panel_checkboxes.values():
            visibility_layout.addWidget(checkbox)
        layout.addWidget(visibility_group)

        # Primary Actions Row
        actions_layout = QHBoxLayout()
        generate_btn = QPushButton("Generate Analysis")
        generate_btn.setDefault(True)
        generate_btn.clicked.connect(self.refresh_plot)
        
        save_btn = QPushButton("Save Figure")
        save_btn.clicked.connect(self.save_current_figure)
        
        actions_layout.addWidget(generate_btn)
        actions_layout.addWidget(save_btn)
        layout.addLayout(actions_layout)

        layout.addStretch() # Pushes layout elegantly up to the top
        scroll_area.setWidget(container)
        return scroll_area

    def _apply_preset(self, value):
        if value:
            self.start_value_spin.setValue(int(value))

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

    def _draw_transition_heatmap(self, ax, parities, block_length):
        if len(parities) < block_length + 1:
            ax.text(0.5, 0.5, "Insufficient step counts for transition maps.", ha="center", va="center", color="#888888")
            return

        blocks = ["".join(str(b) for b in parities[i : i + block_length]) for i in range(len(parities) - block_length + 1)]
        transitions = Counter(zip(blocks[:-1], blocks[1:]))
        labels = [format(i, f"0{block_length}b") for i in range(2**block_length)]
        index = {label: i for i, label in enumerate(labels)}

        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for (src, dst), count in transitions.items():
            if src in index and dst in index:
                matrix[index[src], index[dst]] = count

        image = ax.imshow(matrix, cmap="plasma", aspect="auto")
        ax.set_title(f"Parity Block Transitions (Length: {block_length})", color="#e0e0e0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", color="#b0b0b0")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color="#b0b0b0")
        
        for spine in ax.spines.values():
            spine.set_color("#333333")
            
        self.figure.colorbar(image, ax=ax, shrink=0.9, pad=0.02)

    def _draw_dashboard(self, values, parities, window, block_length):
        self.figure.clear()
        selected_panels = self._selected_panels()

        if not selected_panels:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Enable configurations in display subplots panel.", ha="center", va="center", color="#888888")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        panel_heights = {"orbit": 2.0, "density": 1.0, "blocks": 1.0, "transition": 1.4}
        heights = [panel_heights.get(panel, 1.0) for panel in selected_panels]
        gs = self.figure.add_gridspec(len(selected_panels), 1, height_ratios=heights, hspace=0.4)

        for index, panel in enumerate(selected_panels):
            ax = self.figure.add_subplot(gs[index])
            ax.set_facecolor("#181818")
            ax.tick_params(colors="#b0b0b0", labelsize=8)
            ax.grid(color="#2a2a2a", linestyle=":", linewidth=0.6)

            for spine in ax.spines.values():
                spine.set_color("#333333")

            if panel == "orbit":
                ax.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1.5, color="#3182ce", label="Orbit")
                ax.set_title("Accelerated Orbit and Parity Shadow Sequence", color="#e0e0e0")
                ax.set_ylabel("Numerical Value", color="#b0b0b0")
                
                ax2 = ax.twinx()
                ax2.step(range(len(parities)), parities, where="mid", color="#e53e3e", linewidth=1.2, alpha=0.8)
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(["Even (0)", "Odd (1)"], color="#b0b0b0", fontsize=8)
                ax2.spines["right"].set_color("#333333")

            elif panel == "density":
                if len(parities) >= window:
                    positions = [i + window / 2 for i in range(len(parities) - window + 1)]
                    density = [np.mean(parities[i : i + window]) for i in range(len(parities) - window + 1)]
                    ax.plot(positions, density, linewidth=1.8, color="#dd6b20")
                    ax.fill_between(positions, density, 0, color="#dd6b20", alpha=0.1)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Odd Density", color="#b0b0b0")
                ax.set_title(f"Rolling Parity Density (Window: {window})", color="#e0e0e0")

            elif panel == "blocks":
                blocks = ["".join(str(b) for b in parities[i : i + block_length]) for i in range(len(parities) - block_length + 1)]
                counts = Counter(blocks)
                labels = [format(i, f"0{block_length}b") for i in range(2**block_length)]
                frequencies = [counts.get(label, 0) for label in labels]
                
                ax.bar(labels, frequencies, color="#38a169", alpha=0.85, edgecolor="#2f855a", linewidth=0.7)
                ax.set_title(f"Parity Block Distribution (Length: {block_length})", color="#e0e0e0")
                ax.set_ylabel("Occurrences", color="#b0b0b0")
                ax.set_xticklabels(labels, rotation=45, ha="right")

            elif panel == "transition":
                self._draw_transition_heatmap(ax, parities, block_length)

        self.canvas.draw_idle()

    def _update_summary(self, n, values, parities, window, block_length):
        shadow_word = "".join(str(p) for p in parities)
        odd_count = sum(parities)
        even_count = len(parities) - odd_count
        
        if len(parities) >= block_length:
            blocks = [shadow_word[i : i + block_length] for i in range(len(shadow_word) - block_length + 1)]
            most_common = Counter(blocks).most_common(1)
            block_summary = f"Common Block: {most_common[0][0]} ({most_common[0][1]}x)" if most_common else "n/a"
        else:
            block_summary = "Common Block: n/a"

        summary = (
            f"[Input Vector]: {n} | [Orbit Steps]: {len(values)} | [Parity Composition]: {odd_count} Odd / {even_count} Even\n"
            f"[Bit Sequence]: {shadow_word[:76]}{'...' if len(shadow_word) > 76 else ''}\n"
            f"[Metrics Analysis]: Mode Window={window} | {block_summary}"
        )
        self.summary_label.setText(summary)

    def save_current_figure(self):
        out_dir = os.path.join(os.path.dirname(__file__), "shadow_figures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"collatz_shadow_{self.start_value_spin.value()}.png")
        self.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        QMessageBox.information(self, "Export Success", f"High-res diagnostic saved to:\n{out_path}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    # Clean Dark UI Engine Engine Palette Configuration
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(22, 22, 22))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(49, 130, 206))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    sys_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    sys_font.setPointSize(9)
    app.setFont(sys_font)

    window = CollatzShadowExplorer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
