import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def validate_input(n):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    if n > 10**18:  # Limit to prevent excessive computation
        raise ValueError("Input is too large. Please choose a smaller integer.")

@njit
def collatz_sequence(n):
    """Compute the Collatz sequence for a given number n."""
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

def plot_collatz(n):
    """Generate and plot the Collatz sequence for a given number n."""
    validate_input(n)  # Validate input
    seq = collatz_sequence(n)  # Compute sequence
    steps = np.arange(len(seq))  # X-axis: step index
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, seq, marker='o', linestyle='-', markersize=4, label=f'Collatz({n})')
    
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Collatz Sequence for {n}")
    #plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #plt.legend()
    plt.show()

plot_collatz(27)