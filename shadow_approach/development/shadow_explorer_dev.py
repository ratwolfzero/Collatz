import os
import sys
import math
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
        return [n], [], [], []

    values = [n]
    parities = []
    valuations = []
    energy = []

    x = n
    for _ in range(max_steps):
        if x == 1:
            break
            
        if x % 2 == 0:
            x //= 2
            parities.append(0)
            valuations.append(0)
            energy.append(0)
        else:
            # 1. Calculate the metrics (Valuation & Energy) for (3x + 1)
            y = 3 * x + 1
            v = 0
            temp_y = y
            while temp_y % 2 == 0:
                temp_y //= 2
                v += 1
                
            # 2. Perform the original accelerated step 
            x = y // 2
            
            # 3. Store the synchronized data
            parities.append(1)
            valuations.append(v)
            energy.append(v - math.log2(3))
            
        values.append(x)
        
    return values, parities, valuations, energy


def lz_complexity(sequence):
    vocab = set()
    prefix = ""
    complexity = 0

    for item in sequence:
        prefix += f"{item:.2f}" if isinstance(item, float) else str(item)
        if prefix not in vocab:
            vocab.add(prefix)
            complexity += 1
            prefix = ""

    if prefix:
        complexity += 1

    return complexity


class CollatzShadowExplorer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collatz Symbolic Dynamics Laboratory")
        self.resize(1420, 950)

        self.figure = Figure(dpi=120, facecolor="#1e1e1e", constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._build_ui()
        
        # FIX: Fire the state manager on boot instead of refreshing directly
        # This aligns the checkboxes to the default selection ("Parity Shadow") instantly.
        self._on_mode_changed(self.mode_combo.currentText())

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
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
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

        param_group = QGroupBox("Sequence Controls")
        form_layout = QFormLayout(param_group)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setSpacing(8)

        self.start_value_edit = QLineEdit("27")
        validator = QRegularExpressionValidator(QRegularExpression(r"[1-9]\d*"))
        self.start_value_edit.setValidator(validator)
        form_layout.addRow("Initial Value:", self.start_value_edit)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Parity Shadow", "Valuation Shadow", "Energy Shadow"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        form_layout.addRow("Analysis Mode:", self.mode_combo)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 5000)
        self.max_steps_spin.setValue(2000)
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
            ["27", "97", "871", "6171", "77031", "837799", "8400511", "63728127", "670617279", "93571393692802302", "931386509544713451"]
        )
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        form_layout.addRow("Quick Preset:", self.preset_combo)
        layout.addWidget(param_group)

        visibility_group = QGroupBox("Display Subplots (Max 4)")
        visibility_layout = QVBoxLayout(visibility_group)
        visibility_layout.setSpacing(6)

        self.panel_checkboxes = {
            "orbit": QCheckBox("Orbit + Shadow Track"),
            "phasespace": QCheckBox("Delay Embedding (x_n vs x_n+1)"),
            "recurrence": QCheckBox("Similarity Matrix (2D Texture)"),
            "spectral_entropy": QCheckBox("Rolling Spectral Entropy"),
            "density": QCheckBox("Rolling Event Density"),
            "blocks": QCheckBox("Block Distribution Metric"),
            "turtle": QCheckBox("2D Turtle Walk (Fractal)"),
            "entropy": QCheckBox("Shannon Block Entropy"),
            "lz_complexity": QCheckBox("Lempel-Ziv Complexity"),
            "fft": QCheckBox("Power Spectrum (FFT)"),
            "network": QCheckBox("Observed Transition Network"),
            "autocorr": QCheckBox("Autocorrelation Lag"),
            "heatmap": QCheckBox("Transition Matrix Heatmap"),
            "runlength": QCheckBox("Run-Length Histogram"),
            "walk": QCheckBox("Cumulative Drift (Parity)"),
            "spectrogram": QCheckBox("Contraction Spectrogram (Valuations)"), 
        }

        defaults = ["walk", "spectrogram", "heatmap", "network"]
        for key in defaults:
            if key in self.panel_checkboxes:
                self.panel_checkboxes[key].setChecked(True)

        for checkbox in self.panel_checkboxes.values():
            checkbox.toggled.connect(self._enforce_panel_limit)
            visibility_layout.addWidget(checkbox)

        layout.addWidget(visibility_group)

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

    def _on_mode_changed(self, mode):
        """Dynamically adapts the UI controls based on the selected mode."""
        self._set_panel_state("turtle", enabled=(mode == "Parity Shadow"))
        self._set_panel_state("spectrogram", enabled=(mode == "Valuation Shadow"))
        
        if mode == "Energy Shadow":
            self.panel_checkboxes["walk"].setText("Cumulative Drift (Energy)")
            self._set_panel_state("walk", enabled=True)
        elif mode == "Parity Shadow":
            self.panel_checkboxes["walk"].setText("Cumulative Drift (Parity)")
            self._set_panel_state("walk", enabled=True)
        else:
            self._set_panel_state("walk", enabled=False)

        self.refresh_plot()

    def _set_panel_state(self, key, enabled):
        """Helper to safely handle checkboxes state transformations without triggering cycles."""
        cb = self.panel_checkboxes[key]
        cb.setEnabled(enabled)
        if not enabled:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def _enforce_panel_limit(self, checked):
        if not checked: return
        active_panels = [cb for cb in self.panel_checkboxes.values() if cb.isChecked()]
        if len(active_panels) > 4:
            self.sender().blockSignals(True)
            self.sender().setChecked(False)
            self.sender().blockSignals(False)
            self.summary_label.setText(
                "⚠️ <b>Display Limit Reached:</b> Maximum 4 concurrent visualization streams allowed."
            )

    def _apply_preset(self, value):
        if value:
            self.start_value_edit.setText(value)

    def refresh_plot(self):
        text = self.start_value_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Invalid Input", "Please enter a positive integer.")
            return

        n = int(text)
        max_steps = self.max_steps_spin.value()
        window = self.window_spin.value()
        block_length = self.block_length_spin.value()
        mode = self.mode_combo.currentText()

        values, parities, valuations, energy = collatz_accelerated(n, max_steps=max_steps)
        
        if mode == "Parity Shadow":
            active_seq = parities
        elif mode == "Valuation Shadow":
            active_seq = valuations
        else:
            active_seq = energy

        self._draw_dashboard(values, active_seq, parities, valuations, energy, mode, window, block_length)
        self._update_summary(n, values, active_seq, mode, window, block_length)

    def _selected_panels(self):
        return [name for name, checkbox in self.panel_checkboxes.items() if checkbox.isChecked()]

    # =========================================================================
    # VISUALIZATION ENGINES
    # =========================================================================

    def _draw_phase_space(self, ax, values, sequence):
        if len(values) < 2: return
        x_coords = values[:-1]
        y_coords = values[1:]
        c_vals = sequence[:-1] if len(sequence) == len(values) else sequence[:len(x_coords)]

        ax.plot(x_coords, y_coords, color="#4a5568", linewidth=0.5, alpha=0.4, zorder=1)
        ax.scatter(x_coords, y_coords, c=c_vals, cmap="coolwarm", s=10, alpha=0.75, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Orbit Delay Embedding (x_n vs x_n+1)", color="#e0e0e0")

    def _draw_recurrence_matrix(self, ax, sequence):
        if len(sequence) < 4: return
        p_arr = np.array(sequence).reshape(-1, 1)
        if isinstance(sequence[0], float):
            diff = np.abs(p_arr - p_arr.T)
            recurrence_matrix = (diff < 1e-5).astype(int)
        else:
            recurrence_matrix = (p_arr == p_arr.T).astype(int)

        ax.imshow(recurrence_matrix, cmap="binary", origin="lower", interpolation="none", alpha=0.85)
        ax.set_title("Sequence Similarity Matrix (2D Global Textures)", color="#e0e0e0")

    def _draw_spectral_entropy(self, ax, sequence, window):
        if len(sequence) < window + 4: return
        signal = np.array(sequence, dtype=float)
        
        if set(signal).issubset({0, 1}):
            signal = signal * 2 - 1
            
        se_values, positions = [], []
        for i in range(len(signal) - window + 1):
            chunk = signal[i: i + window]
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
        ax.set_title(f"Rolling Spectral Entropy (Window: {window})", color="#e0e0e0")

    def _draw_observed_nlock_transition_network(self, ax, sequence, block_length):
        if len(sequence) < block_length + 1: return
        
        fmt = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
        blocks = ["-".join(fmt(b) for b in sequence[i: i + block_length]) for i in range(len(sequence) - block_length + 1)]
        transitions = Counter(zip(blocks[:-1], blocks[1:]))

        if len(transitions) > 30:
            transitions = dict(transitions.most_common(30))

        G = nx.DiGraph()
        for (src, dst), weight in transitions.items():
            G.add_edge(src, dst, weight=weight)

        pos = nx.spring_layout(G, seed=42)
        edges = G.edges(data=True)
        weights = [d["weight"] for _, _, d in edges]
        max_w = max(weights) if weights else 1
        normalized_weights = [0.8 + (w / max_w) * 2.5 for w in weights]

        nx.draw_networkx_edges(G, pos, ax=ax, width=normalized_weights, edge_color="#718096", arrows=True, arrowsize=10)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#2b6cb0", node_size=200, edgecolors="#63b3ed")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#f7fafc", font_weight="bold")
        ax.set_title(f"Observed Block Transition Network (Block: {block_length})", color="#e0e0e0")
        ax.axis("off")

    def _draw_turtle_walk(self, ax, parities):
        if not parities: return
        angles = np.where(np.array(parities) == 0, np.pi / 3, -np.pi / 3)
        headings = np.cumsum(angles)
        dx, dy = np.cos(headings), np.sin(headings)
        x = np.concatenate(([0], np.cumsum(dx)))
        y = np.concatenate(([0], np.cumsum(dy)))

        ax.plot(x, y, color="#9f7aea", linewidth=1.5, alpha=0.9)
        ax.scatter([0], [0], color="#48bb78", zorder=5, label="Start")
        ax.scatter([x[-1]], [y[-1]], color="#f56565", zorder=5, label="End")
        ax.set_title("2D Turtle Random Walk (Requires Parity Base)", color="#e0e0e0")
        ax.legend(loc="best", frameon=False, labelcolor="#e0e0e0")
        ax.set_aspect("equal", "datalim")

    def _draw_shannon_entropy(self, ax, sequence, window, block_length):
        if len(sequence) < window + block_length: return
        fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
        
        entropies, positions = [], []
        for i in range(len(sequence) - window + 1):
            chunk = sequence[i: i + window]
            blocks = ["-".join(fmt(b) for b in chunk[j: j + block_length]) for j in range(len(chunk) - block_length + 1)]
            counts = Counter(blocks)
            total = sum(counts.values())
            ent = -sum((count / total) * np.log2(count / total) for count in counts.values())
            entropies.append(ent)
            positions.append(i + window / 2)

        ax.plot(positions, entropies, linewidth=1.8, color="#38b2ac")
        ax.fill_between(positions, entropies, 0, color="#38b2ac", alpha=0.15)
        
        unique_states = len(set(sequence))
        max_ent = block_length * np.log2(unique_states) if unique_states > 0 else block_length
        ax.set_ylim(0, max(max_ent, max(entropies)*1.1) if entropies else block_length)
        ax.set_ylabel("Bits of Entropy", color="#b0b0b0")
        ax.set_title(f"Rolling Shannon Entropy (Window: {window}, Block: {block_length})", color="#e0e0e0")

    def _draw_lz_complexity(self, ax, sequence, window):
        if len(sequence) < window + 2: return
        lz_values, positions = [], []
        N = max(2, window)

        k, bits_used, max_lz = 1, 0, 0
        while bits_used + (k * (2**k)) <= N:
            bits_used += k * (2**k)
            max_lz += 2**k
            k += 1
        max_lz += (N - bits_used) // k

        for i in range(len(sequence) - window + 1):
            chunk = sequence[i: i + window]
            raw_lz = lz_complexity(chunk)
            normalized_lz = raw_lz / max_lz
            lz_values.append(normalized_lz)
            positions.append(i + window / 2)

        ax.plot(positions, lz_values, linewidth=1.8, color="#e53e3e")
        ax.fill_between(positions, lz_values, 0, color="#e53e3e", alpha=0.15)
        ax.axhline(1.0, color="#718096", linestyle="--", linewidth=1.2, alpha=0.8)
        
        y_max = max(1.1, max(lz_values) * 1.1) if lz_values else 1.1
        ax.set_ylim(0, y_max)
        ax.set_title(f"Rolling Lempel-Ziv Complexity (Window: {window})", color="#e0e0e0")

    def _draw_fft(self, ax, sequence):
        if len(sequence) < 4: return
        signal = np.array(sequence, dtype=float)
        if set(signal).issubset({0, 1}):
            signal = signal * 2 - 1
            
        signal -= np.mean(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        power = fft_vals**2
        freqs = np.fft.rfftfreq(len(signal))

        ax.plot(freqs, power, color="#ed8936", linewidth=1.5)
        ax.fill_between(freqs, power, 0, color="#ed8936", alpha=0.15)
        ax.set_title("Power Spectrum (Fast Fourier Transform)", color="#e0e0e0")

    def _draw_autocorrelation(self, ax, sequence):
        if len(sequence) < 5: return
        signal = np.array(sequence, dtype=float)
        signal -= np.mean(signal)
        corr = np.correlate(signal, signal, mode="full")
        corr = corr[len(signal) - 1:]

        if corr[0] == 0:
            ax.text(0.5, 0.5, "Zero variance detected.", ha="center", va="center", color="#888888")
            return

        corr = corr / corr[0]
        ax.plot(corr, color="#63b3ed", linewidth=1.6)
        ax.fill_between(range(len(corr)), corr, 0, alpha=0.15, color="#63b3ed")
        ax.set_title("Autocorrelation (Signal Memory)", color="#e0e0e0")

    def _draw_transition_heatmap(self, ax, sequence):
        if len(sequence) < 2: return
        unique_states = sorted(list(set(sequence)))
        
        if len(unique_states) > 10:
            ax.text(0.5, 0.5, "Sequence too continuous for discrete heatmap.", ha="center", va="center", color="#888888")
            return
            
        state_to_idx = {state: i for i, state in enumerate(unique_states)}
        matrix = np.zeros((len(unique_states), len(unique_states)))
        
        for a, b in zip(sequence[:-1], sequence[1:]):
            matrix[state_to_idx[a], state_to_idx[b]] += 1
            
        if matrix.sum() > 0:
            matrix = matrix / matrix.sum()

        cmap = plt.get_cmap("viridis")
        im = ax.imshow(matrix, cmap=cmap)
        
        fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
        labels = [fmt(s) for s in unique_states]
        
        ax.set_title("Dynamic Transition Matrix", color="#e0e0e0")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        
        for i in range(len(unique_states)):
            for j in range(len(unique_states)):
                value = matrix[i, j]
                normed = im.norm(value)
                r, g, b, _ = cmap(normed)
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "black" if luminance > 0.5 else "white"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)

    def _draw_run_length(self, ax, sequence):
        if len(sequence) < 2: return
        runs, length = [], 1
        current = sequence[0]
        
        for p in sequence[1:]:
            if (isinstance(p, float) and abs(p - current) < 1e-5) or (p == current):
                length += 1
            else:
                runs.append(length)
                current = p
                length = 1
        runs.append(length)
        
        ax.hist(runs, bins=range(1, max(runs) + 2) if runs else 10, color="#ed8936", alpha=0.85, edgecolor="black")
        ax.set_title("Run-Length Distribution Spectrum", color="#e0e0e0")

    def _draw_parity_walk(self, ax, parities):
        if not parities: return
        signal = np.array(parities) * 2 - 1
        walk = np.cumsum(signal)
        ax.plot(walk, color="#9f7aea", linewidth=1.6)
        ax.axhline(0, color="white", alpha=0.4, linewidth=0.8)
        ax.set_title("Cumulative Parity Drift", color="#e0e0e0")
        ax.set_ylabel("Cumulative Drift", color="#b0b0b0")

    def _draw_energy_walk(self, ax, energy):
        if not energy: return
        walk = np.cumsum(energy)
        ax.plot(walk, color="#f6ad55", linewidth=1.5)
        ax.axhline(0, color="white", alpha=0.3, linestyle="--")
        ax.set_title("Cumulative Energy Drift (Contraction Strength)", color="#e0e0e0")
        ax.set_ylabel("Accumulated Energy", color="#b0b0b0")

    def _draw_contraction_spectrogram(self, ax, valuations):
        odd_steps = [i for i, v in enumerate(valuations) if v > 0]
        vals = [v for v in valuations if v > 0]
        if not odd_steps:
            ax.text(0.5, 0.5, "No odd steps found.", ha="center", va="center", color="#888888")
            return
            
        ax.scatter(odd_steps, vals, c=vals, cmap="inferno", s=30, alpha=0.8)
        ax.vlines(odd_steps, 0, vals, color="#4a5568", alpha=0.3)
        ax.set_title("Spectrogram of Contraction Events (Valuations)", color="#e0e0e0")
        ax.set_ylabel("Divisibility Power (v)", color="#b0b0b0")

    def _draw_dashboard(self, values, sequence, parities, valuations, energy, mode, window, block_length):
        self.figure.clear()
        selected_panels = self._selected_panels()

        if not selected_panels:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Enable configurations inside display options grid.", ha="center", va="center", color="#888888")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        panel_heights = {
            "orbit": 2.0, "phasespace": 2.0, "recurrence": 4.0, "spectral_entropy": 1.0,
            "density": 1.0, "blocks": 1.0, "turtle": 2.0, "entropy": 1.0, "lz_complexity": 1.0,
            "fft": 1.0, "network": 2.0, "autocorr": 1.0, "heatmap": 2.0, "runlength": 1.0,
            "walk": 1.5, "spectrogram": 2.0,
        }

        heights = [panel_heights.get(panel, 1.0) for panel in selected_panels]
        gs = self.figure.add_gridspec(len(selected_panels), 1, height_ratios=heights, hspace=0.45)

        for index, panel in enumerate(selected_panels):
            ax = self.figure.add_subplot(gs[index])
            ax.set_facecolor("#181818")
            ax.tick_params(colors="#b0b0b0", labelsize=8)
            ax.grid(color="#2a2a2a", linestyle=":", linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_color("#333333")

            if panel == "orbit":
                ax.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1.5, color="#3182ce")
                ax.set_title(f"Accelerated Orbit and {mode} Overlay", color="#e0e0e0")
                ax.set_ylabel("Numerical Value Space", color="#b0b0b0")
                ax2 = ax.twinx()
                ax2.step(range(len(sequence)), sequence, where="mid", color="#e53e3e", linewidth=1.2, alpha=0.8)
                ax2.tick_params(colors="#b0b0b0", labelsize=8)
                ax2.spines["right"].set_color("#333333")

            elif panel == "phasespace":
                self._draw_phase_space(ax, values, sequence)
            elif panel == "recurrence":
                self._draw_recurrence_matrix(ax, sequence)
            elif panel == "spectral_entropy":
                self._draw_spectral_entropy(ax, sequence, window)
            elif panel == "density":
                if len(sequence) >= window:
                    positions = [i + window / 2 for i in range(len(sequence) - window + 1)]
                    density = [np.mean(sequence[i: i + window]) for i in range(len(sequence) - window + 1)]
                    ax.plot(positions, density, linewidth=1.8, color="#dd6b20")
                    ax.fill_between(positions, density, 0, color="#dd6b20", alpha=0.1)
                ax.set_title(f"Rolling Average Metric (Window: {window})", color="#e0e0e0")
            elif panel == "blocks":
                fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
                blocks = ["-".join(fmt(b) for b in sequence[i: i + block_length]) for i in range(len(sequence) - block_length + 1)]
                counts = Counter(blocks)
                
                common = counts.most_common(15)
                labels = [c[0] for c in common]
                frequencies = [c[1] for c in common]
                x_positions = range(len(labels))
                ax.bar(x_positions, frequencies, color="#38a169", alpha=0.85, edgecolor="#2f855a", linewidth=0.7)
                ax.set_title(f"Top {len(labels)} Block Distribution Metrics (Length: {block_length})", color="#e0e0e0")
                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, rotation=45, ha="right")
            elif panel == "turtle":
                self._draw_turtle_walk(ax, parities)
            elif panel == "entropy":
                self._draw_shannon_entropy(ax, sequence, window, block_length)
            elif panel == "lz_complexity":
                self._draw_lz_complexity(ax, sequence, window)
            elif panel == "fft":
                self._draw_fft(ax, sequence)
            elif panel == "network":
                self._draw_observed_nlock_transition_network(ax, sequence, block_length)
            elif panel == "autocorr":
                self._draw_autocorrelation(ax, sequence)
            elif panel == "heatmap":
                self._draw_transition_heatmap(ax, sequence)
            elif panel == "runlength":
                self._draw_run_length(ax, sequence)
            elif panel == "walk":
                if mode == "Energy Shadow":
                    self._draw_energy_walk(ax, energy)
                elif mode == "Parity Shadow":
                    self._draw_parity_walk(ax, parities)
            elif panel == "spectrogram":
                self._draw_contraction_spectrogram(ax, valuations)

        self.figure.canvas.draw_idle()

    def _update_summary(self, n, values, sequence, mode, window, block_length):
        fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
        shadow_word = " ".join(fmt(p) for p in sequence)
        
        if len(sequence) >= block_length:
            blocks = ["-".join(fmt(b) for b in sequence[i: i + block_length]) for i in range(len(sequence) - block_length + 1)]
            most_common = Counter(blocks).most_common(1)
            block_summary = f"Common Block: {most_common[0][0]} ({most_common[0][1]}x)" if most_common else "n/a"
        else:
            block_summary = "Common Block: n/a"

        summary = (
            f"[Input Value]: {n} | [Total Steps]: {len(values)} | [Active Mode]: {mode}\n"
            f"[Symbolic Sequence]: {shadow_word[:80]}{'...' if len(shadow_word) > 80 else ''}\n"
            f"[System Space Config]: Target Window={window} | {block_summary}"
        )
        self.summary_label.setText(summary)

    def save_current_figure(self):
        out_dir = os.path.join(os.path.dirname(__file__), "shadow_figures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"collatz_{self.mode_combo.currentText().replace(' ', '_').lower()}_{self.start_value_edit.text()}.png")
        self.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        QMessageBox.information(self, "Export Complete", f"High-fidelity matrix profile saved to:\n{out_path}")


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
