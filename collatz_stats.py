import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def collatz_sequence(n):
    """Returns the Collatz sequence starting at n."""
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq

def stopping_time(n):
    """Returns the number of steps to reach 1."""
    count = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        count += 1
    return count

def compute_stopping_times(max_n):
    xs = np.arange(1, max_n + 1)
    ys = np.zeros_like(xs)
    for i, n in enumerate(tqdm(xs, desc="Calculating stopping times")):
        ys[i] = stopping_time(n)
    return xs, ys

def plot_stopping_times(xs, ys):
    plt.figure(figsize=(12, 6))
    plt.scatter(xs, ys, s=1, alpha=0.6, label='Stopping time')
    
    # Highlight n = 27
    n_27_index = np.where(xs == 27)[0][0]
    plt.scatter(xs[n_27_index], ys[n_27_index], color='red', s=40, label='n = 27 (s = {})'.format(ys[n_27_index]))

    plt.title('Collatz Stopping Times for n = 1 to {}'.format(xs[-1]))
    plt.xlabel('Starting number n')
    plt.ylabel('Stopping time (number of steps to reach 1)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_histogram(ys):
    plt.figure(figsize=(10, 4))
    plt.hist(ys, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Stopping Times')
    plt.xlabel('Stopping time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_moving_average(xs, ys, window=100):
    moving_avg = np.convolve(ys, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12, 5))
    plt.plot(xs[window-1:], moving_avg, color='orange', label=f'Moving average (window={window})')
    plt.title('Moving Average of Stopping Times')
    plt.xlabel('n')
    plt.ylabel('Average stopping time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    max_n = 10000  # You can increase this to 100000 or more
    xs, ys = compute_stopping_times(max_n)

    plot_stopping_times(xs, ys)
    plot_histogram(ys)
    plot_moving_average(xs, ys)




