import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import windows, lfilter
from numpy.random import default_rng

def compute_fft_beta(signal, sample_spacing=1.0, window_type=None, plot_label=None):
    n = len(signal)
    signal = signal - np.mean(signal)
    signal = signal / np.std(signal)

    if window_type == 'hann':
        win = windows.hann(n)
        signal *= win
    elif window_type == 'hamming':
        win = windows.hamming(n)
        signal *= win
    elif window_type == 'blackman':
        win = windows.blackman(n)
        signal *= win
    # else: rectangular window (no windowing)

    yf = fft(signal)
    xf = fftfreq(n, sample_spacing)[:n//2]
    power = np.abs(yf[:n//2])**2

    # Fit power-law to the log-log power spectrum
    mask = (xf > 0) & (power > 0)
    log_x = np.log(xf[mask])
    log_y = np.log(power[mask])
    coeffs = np.polyfit(log_x, log_y, 1)
    beta = -coeffs[0]

    # Optional plot
    plt.figure(figsize=(8, 4))
    plt.loglog(xf[1:], power[1:], label=f'{plot_label} (β ≈ {beta:.2f})')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title(f'Power Spectrum - {plot_label}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return beta

def generate_white_noise(n):
    return np.random.randn(n)

def generate_brownian_motion(n):
    return np.cumsum(np.random.randn(n))

def generate_pink_noise(n):
    # Voss-McCartney approximation using IIR filter
    white = np.random.randn(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white)
    return pink

def generate_weierstrass(x, a=0.6, b=3.0, num_terms=50):
    w = np.zeros_like(x)
    for k in range(num_terms):
        w += (a ** k) * np.cos((b ** k) * np.pi * x)
    w -= np.mean(w)
    w /= np.std(w)
    return w

# Main simulation
if __name__ == "__main__":
    n = 2**14  # 16384
    x = np.linspace(0, 1, n)

    # Generate signals
    white = generate_white_noise(n)
    pink = generate_pink_noise(n)
    brownian = generate_brownian_motion(n)
    weier = generate_weierstrass(x, a=0.6, b=3.5, num_terms=100)

    # Compute FFT and power-law exponents
    beta_white = compute_fft_beta(white, window_type='hann', plot_label='White Noise')
    beta_pink = compute_fft_beta(pink, window_type='hann', plot_label='Pink Noise')
    beta_brownian = compute_fft_beta(brownian, window_type='hann', plot_label='Brownian Motion')
    beta_weier = compute_fft_beta(weier, window_type='hann', plot_label='Weierstrass Function')

    # Print results
    print(f"White noise β ≈ {beta_white:.2f}")
    print(f"Pink noise β ≈ {beta_pink:.2f}")
    print(f"Brownian motion β ≈ {beta_brownian:.2f}")
    print(f"Weierstrass function β ≈ {beta_weier:.2f}")
