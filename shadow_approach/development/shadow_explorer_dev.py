import os
import sys
from collections import Counter
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtCore import QRegularExpression
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
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
    QLineEdit,
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


def collatz_accelerated(n, max_steps=1000):
    if n <= 0:
        return [n], []

    values = [n]
    parities = []
    x = n
    for _ in range(max_steps):
        if x == 1:
            break
        if x % 2 == 0:
            x //= 2
            parities.append(0)
        else:
            x = (3 * x + 1) // 2
            parities.append(1)
        values.append(x)
    return values, parities


def lz_complexity(sequence):
    """
    Calculates dictionary-based Lempel-Ziv (LZ78) complexity.
    Scans the sequence and counts the number of unique, non-overlapping 
    vocabulary patterns required to construct it.
    """
    vocab = set()
    prefix = ""
    complexity = 0

    for item in sequence:
        prefix += str(item)
        if prefix not in vocab:
            vocab.add(prefix)
            complexity += 1
            prefix = ""

    # Add 1 for the final trailing sequence if it didn't complete a new word
    if prefix:
        complexity += 1

    return complexity


class CollatzShadowExplorer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collatz Symbolic Dynamics Laboratory")
        self.resize(1420, 950)

        # Core Matplotlib Setup
        self.figure = Figure(dpi=120, facecolor="#1e1e1e",
                             constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._build_ui()
        self.refresh_plot()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        plot_panel = self._create_plot_panel()
        control_panel = self._create_control_panel()

        splitter.addWidget(plot_panel)
        splitter.addWidget(control_panel)
        splitter.setSizes([1040, 340])

    def _create_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.summary_label = QLabel("Initializing analysis...")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.summary_label.setMinimumHeight(70)
        self.summary_label.setStyleSheet(
            "QLabel { background-color: #1e1e1e; border: 1px solid #333333; border-radius: 6px; padding: 10px; "
            "font-family: 'Menlo', 'Courier New', monospace; }"
        )
        layout.addWidget(self.summary_label)

        plot_box = QGroupBox("Analysis Dashboard")
        plot_box_layout = QVBoxLayout(plot_box)
        plot_box_layout.setContentsMargins(8, 8, 8, 8)

        plot_box_layout.addWidget(self.toolbar)
        plot_box_layout.addWidget(self.canvas)

        layout.addWidget(plot_box, stretch=1)
        return panel

    def _create_control_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(14)

        # Parameters
        param_group = QGroupBox("Sequence Controls")
        form_layout = QFormLayout(param_group)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setSpacing(8)

        self.start_value_edit = QLineEdit("27")

        # Allow only positive integers
        validator = QRegularExpressionValidator(
            QRegularExpression(r"[1-9]\d*"))
        self.start_value_edit.setValidator(validator)

        form_layout.addRow("Initial Value:", self.start_value_edit)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 10000)
        self.max_steps_spin.setValue(1000)
        form_layout.addRow("Max Steps:", self.max_steps_spin)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(4, 2000)
        self.window_spin.setValue(30)
        form_layout.addRow("Rolling Window:", self.window_spin)

        self.block_length_spin = QSpinBox()
        self.block_length_spin.setRange(2, 8)
        self.block_length_spin.setValue(3)
        form_layout.addRow("Block Length:", self.block_length_spin)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ["27", "97", "871", "6171", "77031", "837799", "8400511", "63728127", "670617279", "93571393692802302"])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        form_layout.addRow("Quick Preset:", self.preset_combo)
        layout.addWidget(param_group)

        # Visibility Group with Max-4 Constraint
        visibility_group = QGroupBox("Display Subplots (Max 4)")
        visibility_layout = QVBoxLayout(visibility_group)
        visibility_layout.setSpacing(6)

        self.panel_checkboxes = {
            "orbit": QCheckBox("Orbit + Shadow Track"),
            "phasespace": QCheckBox("Delay Embedding (x_n vs x_n+1)"),
            "recurrence": QCheckBox("Parity Similarity Matrix (2D)"),
            "spectral_entropy": QCheckBox("Rolling Spectral Entropy"),
            "density": QCheckBox("Rolling Parity Density"),
            "blocks": QCheckBox("Block Recurrence Metric"),
            "turtle": QCheckBox("2D Turtle Walk (Fractal)"),
            "entropy": QCheckBox("Shannon Block Entropy"),
            "lz_complexity": QCheckBox("Lempel-Ziv Algorithmic Complexity"),
            "fft": QCheckBox("Power Spectrum (FFT)"),
            "network": QCheckBox("Observed Block Transition Network"),
            "autocorr": QCheckBox("Autocorrelation Lag"),
            "heatmap": QCheckBox("Transition Matrix Heatmap"),
            "runlength": QCheckBox("Run-Length Histogram"),
            "walk": QCheckBox("Cumulative Parity Drift"),
        }

        # Loaded with the default setup
        defaults = ["orbit", "phasespace", "heatmap", "spectral_entropy"]
        for key in defaults:
            if key in self.panel_checkboxes:
                self.panel_checkboxes[key].setChecked(True)

        for checkbox in self.panel_checkboxes.values():
            checkbox.toggled.connect(self._enforce_panel_limit)
            visibility_layout.addWidget(checkbox)

        layout.addWidget(visibility_group)

        # Actions
        actions_layout = QHBoxLayout()
        generate_btn = QPushButton("Compute Lab")
        generate_btn.setDefault(True)
        generate_btn.clicked.connect(self.refresh_plot)

        save_btn = QPushButton("Save Map")
        save_btn.clicked.connect(self.save_current_figure)

        actions_layout.addWidget(generate_btn)
        actions_layout.addWidget(save_btn)
        layout.addLayout(actions_layout)

        layout.addStretch()
        scroll_area.setWidget(container)
        return scroll_area

    def _enforce_panel_limit(self, checked):
        if not checked:
            return

        active_panels = [
            cb for cb in self.panel_checkboxes.values() if cb.isChecked()]
        if len(active_panels) > 4:
            self.sender().blockSignals(True)
            self.sender().setChecked(False)
            self.sender().blockSignals(False)
            self.summary_label.setText(
                "⚠️ <b>Display Limit Reached:</b> Maximum 4 concurrent visualization streams allowed to prevent window layout collapse."
            )

    def _apply_preset(self, value):
        if value:
            self.start_value_edit.setText(value)

    def refresh_plot(self):
        text = self.start_value_edit.text().strip()

        if not text:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a positive integer."
            )
            return   # <-- MUST be inside the if block

        n = int(text)

        max_steps = self.max_steps_spin.value()
        window = self.window_spin.value()
        block_length = self.block_length_spin.value()

        values, parities = collatz_accelerated(n, max_steps=max_steps)

        self._draw_dashboard(values, parities, window, block_length)
        self._update_summary(n, values, parities, window, block_length)

        n = int(text)
        max_steps = self.max_steps_spin.value()
        window = self.window_spin.value()
        block_length = self.block_length_spin.value()

        values, parities = collatz_accelerated(n, max_steps=max_steps)
        self._draw_dashboard(values, parities, window=window,
                             block_length=block_length)
        self._update_summary(n, values, parities,
                             window=window, block_length=block_length)

    def _selected_panels(self):
        return [name for name, checkbox in self.panel_checkboxes.items() if checkbox.isChecked()]

    # =========================================================================
    # ADVANCED SIGNAL & CHAOS PROCESSING VISUALIZATIONS
    # =========================================================================

    def _draw_phase_space(self, ax, values, parities):
        if len(values) < 2:
            ax.text(0.5, 0.5, "Insufficient depth for mapping.",
                    ha="center", va="center", color="#888888")
            return

        x_coords = values[:-1]
        y_coords = values[1:]

        ax.plot(x_coords, y_coords, color="#4a5568",
                linewidth=0.5, alpha=0.4, zorder=1)
        scatter = ax.scatter(x_coords, y_coords, c=parities,
                             cmap="coolwarm", s=10, alpha=0.75, zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            "Orbit Delay Embedding (x_n vs x_n+1)", color="#e0e0e0")
        ax.set_xlabel("State space x_n (Log)", color="#b0b0b0")
        ax.set_ylabel("State space x_n+1 (Log)", color="#b0b0b0")

    def _draw_recurrence_matrix(self, ax, parities):
        if len(parities) < 4:
            ax.text(0.5, 0.5, "Insufficient matrix depth.",
                    ha="center", va="center", color="#888888")
            return

        p_arr = np.array(parities).reshape(-1, 1)
        recurrence_matrix = (p_arr == p_arr.T).astype(int)

        ax.imshow(recurrence_matrix, cmap="binary",
                  origin="lower", interpolation="none", alpha=0.85)
        ax.set_title(
            "Parity Similarity Matrix (2D Global Textures)", color="#e0e0e0")
        ax.set_xlabel("Sequence Vector Time (i)", color="#b0b0b0")
        ax.set_ylabel("Sequence Vector Time (j)", color="#b0b0b0")
        ax.grid(False)

    def _draw_spectral_entropy(self, ax, parities, window):
        if len(parities) < window + 4:
            ax.text(
                0.5,
                0.5,
                f"Orbit length shorter than tracking window ({window}).",
                ha="center",
                va="center",
                color="#888888",
            )
            return

        # Map parities to balanced signal dynamics (-1, 1)
        signal = np.array(parities, dtype=float) * 2 - 1
        se_values = []
        positions = []

        for i in range(len(signal) - window + 1):
            chunk = signal[i: i + window]
            # Mean-center the chunk
            chunk -= np.mean(chunk)

            fft_vals = np.abs(np.fft.rfft(chunk))
            psd = fft_vals**2
            psd_sum = np.sum(psd)

            if psd_sum > 0:
                psd_norm = psd / psd_sum
                psd_norm = psd_norm[psd_norm > 0]
                entropy = -np.sum(psd_norm * np.log2(psd_norm))

                n_bins = len(fft_vals)
                if n_bins > 1:
                    entropy /= np.log2(n_bins)
            else:
                entropy = 0

            se_values.append(entropy)
            positions.append(i + window / 2)

        ax.plot(positions, se_values, color="#fc8181", linewidth=1.6)
        ax.fill_between(positions, se_values, 0, color="#fc8181", alpha=0.15)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Rolling Spectral Entropy Estimate (Window: {window})", color="#e0e0e0")
        ax.set_ylabel("Normalized Spectral Entropy", color="#b0b0b0")

    # =========================================================================
    # PORTED SUBPLOT VISUALIZATION SUITE
    # =========================================================================

    def _draw_de_bruijn_network(self, ax, parities, block_length):
        if len(parities) < block_length + 1:
            ax.text(0.5, 0.5, "Insufficient step counts for maps.",
                    ha="center", va="center", color="#888888")
            return

        blocks = [
            "".join(str(b) for b in parities[i: i + block_length]) for i in range(len(parities) - block_length + 1)
        ]
        transitions = Counter(zip(blocks[:-1], blocks[1:]))

        G = nx.DiGraph()
        for (src, dst), weight in transitions.items():
            G.add_edge(src, dst, weight=weight)

        pos = nx.spring_layout(G, seed=42)
        edges = G.edges(data=True)
        weights = [d["weight"] for _, _, d in edges]
        max_w = max(weights) if weights else 1
        normalized_weights = [0.8 + (w / max_w) * 2.5 for w in weights]

        nx.draw_networkx_edges(G, pos, ax=ax, width=normalized_weights,
                               edge_color="#718096", arrows=True, arrowsize=12)
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color="#2b6cb0", node_size=350, edgecolors="#63b3ed")
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=8, font_color="#f7fafc", font_weight="bold")
        ax.set_title(
            f"Observed Block Transition Network (Block Length: {block_length})", color="#e0e0e0")
        ax.axis("off")

    def _draw_turtle_walk(self, ax, parities):
        if not parities:
            return
        angles = np.where(np.array(parities) == 0, np.pi / 3, -np.pi / 3)
        headings = np.cumsum(angles)
        dx, dy = np.cos(headings), np.sin(headings)
        x = np.concatenate(([0], np.cumsum(dx)))
        y = np.concatenate(([0], np.cumsum(dy)))

        ax.plot(x, y, color="#9f7aea", linewidth=1.5, alpha=0.9)
        ax.scatter([0], [0], color="#48bb78", zorder=5, label="Start")
        ax.scatter([x[-1]], [y[-1]], color="#f56565", zorder=5, label="End")
        ax.set_title("2D Turtle Random Walk (0=Left, 1=Right)",
                     color="#e0e0e0")
        ax.legend(loc="best", frameon=False, labelcolor="#e0e0e0")
        ax.set_aspect("equal", "datalim")

    def _draw_shannon_entropy(self, ax, parities, window, block_length):
        if len(parities) < window + block_length:
            ax.text(0.5, 0.5, f"Window ({window}) too large.",
                    ha="center", va="center", color="#888888")
            return
        entropies, positions = [], []
        for i in range(len(parities) - window + 1):
            chunk = parities[i: i + window]
            blocks = [
                "".join(str(b) for b in chunk[j: j + block_length]) for j in range(len(chunk) - block_length + 1)
            ]
            counts = Counter(blocks)
            total = sum(counts.values())
            ent = -sum((count / total) * np.log2(count / total)
                       for count in counts.values())
            entropies.append(ent)
            positions.append(i + window / 2)

        ax.plot(positions, entropies, linewidth=1.8, color="#38b2ac")
        ax.fill_between(positions, entropies, 0, color="#38b2ac", alpha=0.15)
        ax.set_ylim(0, block_length)
        ax.set_ylabel("Bits of Entropy", color="#b0b0b0")
        ax.set_title(
            f"Rolling Shannon Entropy (Window: {window}, Block: {block_length})", color="#e0e0e0")

    def _draw_lz_complexity(self, ax, parities, window):
        if len(parities) < window + 2:
            ax.text(0.5, 0.5, "Orbit too short for LZ complexity.",
                    ha="center", va="center", color="#888888")
            return

        lz_values = []
        positions = []
        N = max(2, window)

        # 1. Compute the EXACT maximum possible LZ78 phrases for finite N
        k = 1
        bits_used = 0
        max_lz = 0
        while bits_used + (k * (2**k)) <= N:
            bits_used += k * (2**k)
            max_lz += 2**k
            k += 1
        # Add any remaining fractional phrases from the leftover bits
        max_lz += (N - bits_used) // k

        # 2. Compute rolling window values
        for i in range(len(parities) - window + 1):
            chunk = parities[i: i + window]
            raw_lz = lz_complexity(chunk)

            # Normalize against the true physical upper bound
            normalized_lz = raw_lz / max_lz
            lz_values.append(normalized_lz)
            positions.append(i + window / 2)

        # 3. Plotting Updates
        ax.plot(positions, lz_values, linewidth=1.8, color="#e53e3e")
        ax.fill_between(positions, lz_values, 0, color="#e53e3e", alpha=0.15)

        # A value of 1.0 now strictly means "maximum possible entropy/disorder"
        ax.axhline(1.0, color="#718096", linestyle="--", linewidth=1.2,
                   alpha=0.8, label="Theoretical Randomness Limit")

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Normalized Complexity", color="#b0b0b0")
        ax.set_title(
            f"Rolling Lempel-Ziv Algorithmic Complexity (Window: {window})", color="#e0e0e0")
        ax.legend(loc="upper right", frameon=False,
                  labelcolor="#e0e0e0", fontsize=8)

    def _draw_fft(self, ax, parities):
        if len(parities) < 4:
            return
        signal = np.array(parities, dtype=float) * 2 - 1
        # Remove DC component
        signal -= np.mean(signal)

        fft_vals = np.abs(np.fft.rfft(signal))
        power = fft_vals**2
        freqs = np.fft.rfftfreq(len(signal))

        ax.plot(freqs, power, color="#ed8936", linewidth=1.5)
        ax.fill_between(freqs, power, 0, color="#ed8936", alpha=0.15)
        ax.set_ylabel("Power", color="#b0b0b0")
        ax.set_xlabel("Frequency", color="#b0b0b0")
        ax.set_title("Power Spectrum (Fast Fourier Transform)",
                     color="#e0e0e0")

    def _draw_autocorrelation(self, ax, parities):
        if len(parities) < 5:
            return
        signal = np.array(parities, dtype=float) * 2 - 1
        # Remove DC component
        signal -= np.mean(signal)

        corr = np.correlate(signal, signal, mode="full")
        corr = corr[len(signal) - 1:]
        corr = corr / corr[0]

        ax.plot(corr, color="#63b3ed", linewidth=1.6)
        ax.fill_between(range(len(corr)), corr, 0, alpha=0.15, color="#63b3ed")
        ax.set_title("Autocorrelation (Parity Memory)", color="#e0e0e0")
        ax.set_xlabel("Lag Offset", color="#b0b0b0")
        ax.set_ylabel("Correlation Rate", color="#b0b0b0")

    def _text_color_for_value(self, cmap, norm, value):
        normed = norm(value)
        r, g, b, _ = cmap(normed)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.5 else "white"

    def _draw_transition_heatmap(self, ax, parities):
        if len(parities) < 2:
            return
        matrix = np.zeros((2, 2))
        for a, b in zip(parities[:-1], parities[1:]):
            matrix[a, b] += 1
        if matrix.sum() > 0:
            matrix = matrix / matrix.sum()

        cmap = plt.get_cmap("viridis")
        im = ax.imshow(matrix, cmap=cmap)
        norm = im.norm

        ax.set_title("Parity Transition Matrix", color="#e0e0e0")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0 (Even)", "1 (Odd)"])
        ax.set_yticklabels(["0 (Even)", "1 (Odd)"])
        ax.grid(False)

        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                text_color = self._text_color_for_value(cmap, norm, value)
                ax.text(j, i, f"{value:.2f}", ha="center",
                        va="center", color=text_color)

    def _draw_run_length(self, ax, parities):
        if len(parities) < 2:
            return
        runs = []
        current = parities[0]
        length = 1
        for p in parities[1:]:
            if p == current:
                length += 1
            else:
                runs.append(length)
                current = p
                length = 1
        runs.append(length)
        ax.hist(
            runs, bins=range(1, max(runs) + 2) if runs else 10, color="#ed8936", alpha=0.85, edgecolor="black"
        )
        ax.set_title("Run-Length Distribution Spectrum", color="#e0e0e0")
        ax.set_xlabel("Run Depth", color="#b0b0b0")
        ax.set_ylabel("Frequency", color="#b0b0b0")

    def _draw_parity_walk(self, ax, parities):
        if not parities:
            return
        signal = np.array(parities) * 2 - 1
        walk = np.cumsum(signal)
        ax.plot(walk, color="#9f7aea", linewidth=1.6)
        ax.axhline(0, color="white", alpha=0.4, linewidth=0.8)
        ax.set_title("Cumulative Parity Drift", color="#e0e0e0")
        ax.set_xlabel("Orbit Step", color="#b0b0b0")
        ax.set_ylabel("Cumulative Parity Drift", color="#b0b0b0")

    def _draw_dashboard(self, values, parities, window, block_length):
        self.figure.clear()
        selected_panels = self._selected_panels()

        if not selected_panels:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Enable configurations inside display options grid.",
                ha="center",
                va="center",
                color="#888888",
            )
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        panel_heights = {
            "orbit": 2.0,
            "phasespace": 2.0,
            "recurrence": 4.0,
            "spectral_entropy": 1.0,
            "density": 1.0,
            "blocks": 1.0,
            "turtle": 2.0,
            "entropy": 1.0,
            "lz_complexity": 1.0,
            "fft": 1.0,
            "network": 2.0,
            "autocorr": 1.0,
            "heatmap": 2.0,
            "runlength": 1.0,
            "walk": 1.0,
        }

        heights = [panel_heights.get(panel, 1.0) for panel in selected_panels]
        gs = self.figure.add_gridspec(
            len(selected_panels), 1, height_ratios=heights, hspace=0.45)

        for index, panel in enumerate(selected_panels):
            ax = self.figure.add_subplot(gs[index])
            ax.set_facecolor("#181818")
            ax.tick_params(colors="#b0b0b0", labelsize=8)
            ax.grid(color="#2a2a2a", linestyle=":", linewidth=0.6)

            for spine in ax.spines.values():
                spine.set_color("#333333")

            if panel == "orbit":
                ax.plot(range(len(values)), values, marker="o",
                        markersize=3, linewidth=1.5, color="#3182ce")
                ax.set_title(
                    "Accelerated Orbit and Parity Shadow Sequence", color="#e0e0e0")
                ax.set_ylabel("Numerical Value Space", color="#b0b0b0")

                ax2 = ax.twinx()
                ax2.step(range(len(parities)), parities, where="mid",
                         color="#e53e3e", linewidth=1.2, alpha=0.8)
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(["Even (0)", "Odd (1)"],
                                    color="#b0b0b0", fontsize=8)
                ax2.spines["right"].set_color("#333333")

            elif panel == "phasespace":
                self._draw_phase_space(ax, values, parities)

            elif panel == "recurrence":
                self._draw_recurrence_matrix(ax, parities)

            elif panel == "spectral_entropy":
                self._draw_spectral_entropy(ax, parities, window)

            elif panel == "density":
                if len(parities) >= window:
                    positions = [i + window /
                                 2 for i in range(len(parities) - window + 1)]
                    density = [np.mean(parities[i: i + window])
                               for i in range(len(parities) - window + 1)]
                    ax.plot(positions, density, linewidth=1.8, color="#dd6b20")
                    ax.fill_between(positions, density, 0,
                                    color="#dd6b20", alpha=0.1)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Odd Vector Density", color="#b0b0b0")
                ax.set_title(
                    f"Rolling Parity Density Metrics (Window: {window})", color="#e0e0e0")

            elif panel == "blocks":
                blocks = [
                    "".join(str(b) for b in parities[i: i + block_length])
                    for i in range(len(parities) - block_length + 1)
                ]
                counts = Counter(blocks)
                labels = [format(i, f"0{block_length}b")
                          for i in range(2**block_length)]
                frequencies = [counts.get(label, 0) for label in labels]

                x_positions = range(len(labels))

                ax.bar(x_positions, frequencies, color="#38a169",
                       alpha=0.85, edgecolor="#2f855a", linewidth=0.7)
                ax.set_title(
                    f"Parity Block Distribution Metrics (Length: {block_length})", color="#e0e0e0")
                ax.set_ylabel("Occurrences Index", color="#b0b0b0")

                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, rotation=45, ha="right")

            elif panel == "turtle":
                self._draw_turtle_walk(ax, parities)

            elif panel == "entropy":
                self._draw_shannon_entropy(ax, parities, window, block_length)

            elif panel == "lz_complexity":
                self._draw_lz_complexity(ax, parities, window)

            elif panel == "fft":
                self._draw_fft(ax, parities)

            elif panel == "network":
                self._draw_de_bruijn_network(ax, parities, block_length)

            elif panel == "autocorr":
                self._draw_autocorrelation(ax, parities)

            elif panel == "heatmap":
                self._draw_transition_heatmap(ax, parities)

            elif panel == "runlength":
                self._draw_run_length(ax, parities)

            elif panel == "walk":
                self._draw_parity_walk(ax, parities)

        self.figure.canvas.draw_idle()

    def _update_summary(self, n, values, parities, window, block_length):
        shadow_word = "".join(str(p) for p in parities)
        odd_count = sum(parities)
        even_count = len(parities) - odd_count

        if len(parities) >= block_length:
            blocks = [shadow_word[i: i + block_length]
                      for i in range(len(shadow_word) - block_length + 1)]
            most_common = Counter(blocks).most_common(1)
            block_summary = f"Common Block: {most_common[0][0]} ({most_common[0][1]}x)" if most_common else "n/a"
        else:
            block_summary = "Common Block: n/a"

        summary = (
            f"[Input Value]: {n} | [Total Steps]: {len(values)} | [Parity Splits]: {odd_count} Odd / {even_count} Even\n"
            f"[Raw Parity Bits]: {shadow_word[:76]}{'...' if len(shadow_word) > 76 else ''}\n"
            f"[System Space Configurations]: Target Window={window} | {block_summary}"
        )
        self.summary_label.setText(summary)

    def save_current_figure(self):
        out_dir = os.path.join(os.path.dirname(__file__), "shadow_figures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"collatz_dynamics_matrix_{self.start_value_edit.text()}.png")
        self.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        QMessageBox.information(
            self, "Export Complete", f"High-fidelity matrix profile saved to:\n{out_path}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

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
    sys_font.setPointSize(13)
    app.setFont(sys_font)

    window = CollatzShadowExplorer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
