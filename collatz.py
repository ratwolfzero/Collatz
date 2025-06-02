import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numba import njit

# --- Input Validation ---
def validate_input(n):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    if n > 10**30:
        raise ValueError("Input too large. Please choose a smaller integer.")

# --- Fast Numba-accelerated Collatz sequence for 64-bit ints ---
@njit
def collatz_sequence_fast(n):
    sequence = []
    while n != 1:
        sequence.append(n)
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    sequence.append(1)
    return sequence

# --- Pure Python Collatz sequence for big integers ---
def collatz_sequence_py(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# --- Main Plotting Function ---
def plot_collatz(n, use_log_scale=False):
    validate_input(n)

    try:
        if n <= np.iinfo(np.int64).max:
            seq = collatz_sequence_fast(n)
        else:
            seq = collatz_sequence_py(n)
    except Exception as e:
        raise RuntimeError(f"Failed to compute Collatz sequence: {e}")

    steps = np.arange(len(seq))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, seq, marker='o', linestyle='-', markersize=4, label=f'Collatz({n})')

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Collatz Sequence for {n}")
    if use_log_scale:
        plt.yscale('log')

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- Enforce integer x-ticks ---
    ax = plt.gca()
    max_ticks = 100  # Set threshold for using all integer ticks
    if len(steps) <= max_ticks:
        plt.xticks(steps)  # show all integer steps
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # fewer integer ticks

    plt.tight_layout()
    plt.show()

# --- Main with Query ---
if __name__ == "__main__":
    try:
        user_input = input("Enter a positive integer for the Collatz sequence: ")
        n = int(user_input.strip())

        scale_input = input("Use logarithmic y-scale? (y/n): ").strip().lower()
        use_log = scale_input == 'y'

        plot_collatz(n, use_log_scale=use_log)
    except ValueError as e:
        print(f"Error: {e}")
