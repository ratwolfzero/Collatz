import os
import matplotlib.pyplot as plt
import numpy as np


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


def make_plot_1(values, parities, out_path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(range(len(values)), values, marker='o', linewidth=2, color='#1f77b4')
    ax.set_title('Accelerated Collatz orbit')
    ax.set_xlabel('Accelerated step')
    ax.set_ylabel('Value')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plot_2(values, parities, out_path):
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.step(range(len(parities)), parities, where='mid', color='#d62728', linewidth=2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['even', 'odd'])
    ax.set_title('Parity shadow')
    ax.set_xlabel('Accelerated step')
    ax.set_ylabel('Parity (0 = even, 1 = odd)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plot_3(values, parities, out_path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(range(len(values)), values, marker='o', linewidth=2, color='#2ca02c', label='Orbit')
    ax2 = ax.twinx()
    ax2.step(range(len(parities)), parities, where='mid', color='#d62728', linewidth=1.5, alpha=0.8, label='Parity shadow')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['even', 'odd'])
    ax.set_title('Orbit and parity shadow together')
    ax.set_xlabel('Accelerated step')
    ax.set_ylabel('Value')
    ax2.set_ylabel('Parity (0 = even, 1 = odd)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plot_4(values, parities, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(values)), values, marker='o', linewidth=2, color='#9467bd')
    ax.set_title('Illustrative example: accelerated orbit from 27')
    ax.set_xlabel('Accelerated step')
    ax.set_ylabel('Value')
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.step(range(len(parities)), parities, where='mid', color='#d62728', linewidth=1.5, alpha=0.8)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['even', 'odd'])
    ax2.set_ylabel('Parity (0 = even, 1 = odd)')

    ax.text(0.02, 0.96, 'The parity word is the symbolic shadow of the orbit.',
            transform=ax.transAxes, va='top', ha='left', fontsize=10, style='italic')
    ax.text(0.98, 0.96, 'Accelerated steps shown here',
            transform=ax.transAxes, va='top', ha='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    out_dir = 'shadow_figures'
    os.makedirs(out_dir, exist_ok=True)

    n = 27
    values, parities = collatz_accelerated(n, max_steps=200)

    make_plot_1(values, parities, os.path.join(out_dir, 'orbit.png'))
    make_plot_2(values, parities, os.path.join(out_dir, 'parity_shadow.png'))
    make_plot_3(values, parities, os.path.join(out_dir, 'combined.png'))
    make_plot_4(values, parities, os.path.join(out_dir, 'orbit_plus_parity.png'))

    print(f'Generated figures in {out_dir}')


if __name__ == '__main__':
    main()
