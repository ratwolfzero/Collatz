import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import windows
from tqdm import tqdm

# --- Your existing functions (collatz_sequence, stopping_time, compute_stopping_times, plot_stopping_times, plot_histogram) ---

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
    plt.scatter(xs, ys, s=1, alpha=0.6, color='red', label='Stopping time')
    
    n_27_index = np.where(xs == 27)[0]
    if len(n_27_index) > 0: # Check if 27 is in the range
        plt.scatter(xs[n_27_index], ys[n_27_index], color='blue', s=20, label='n = 27 (s = {})'.format(ys[n_27_index][0]))

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
    plt.scatter(xs, ys, s=1, alpha=0.6, color='red', label='Stopping time')
    
    n_27_index = np.where(xs == 27)[0]
    if len(n_27_index) > 0: # Check if 27 is in the range
        plt.scatter(xs[n_27_index], ys[n_27_index], color='blue', s=20, label='n = 27 (s = {})'.format(ys[n_27_index][0]))

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

# --- Weierstrass function ---
def weierstrass_function(x, a=0.5, b=3, num_terms=100):
    """
    Generates values for the Weierstrass function.
    x: A numpy array of values for which to compute W(x).
    a, b: Parameters for the Weierstrass function.
    num_terms: Number of terms in the infinite sum (approximation).
    """
    result = np.zeros_like(x, dtype=float)
    for k in range(num_terms):
        result += (a**k) * np.cos((b**k) * np.pi * x)
    return result

# --- Combined plot for moving average and Weierstrass ---
def plot_moving_average_with_weierstrass(xs, ys, window=100, weierstrass_a=0.5, weierstrass_b=3, weierstrass_terms=100):
    moving_avg = np.convolve(ys, np.ones(window)/window, mode='valid')
    
    x_weierstrass_domain = np.linspace(0, 1, len(moving_avg)) 
    weierstrass_values = weierstrass_function(x_weierstrass_domain, a=weierstrass_a, b=weierstrass_b, num_terms=weierstrass_terms)

    weierstrass_values_norm = (weierstrass_values - np.mean(weierstrass_values)) / np.std(weierstrass_values)
    
    std_moving_avg = np.std(moving_avg) if np.std(moving_avg) > 1e-9 else 1.0
    weierstrass_scaled = weierstrass_values_norm * std_moving_avg * 0.5 
    weierstrass_shifted = weierstrass_scaled + np.mean(moving_avg)

    plt.figure(figsize=(14, 7))
    plt.plot(xs[window-1:], moving_avg, color='orange', label=f'Collatz Moving Average (window={window})')
    plt.plot(xs[window-1:], weierstrass_shifted, color='purple', linestyle='--', alpha=0.7, 
             label=f'Scaled Weierstrass Function (a={weierstrass_a}, b={weierstrass_b}, terms={weierstrass_terms})')
    
    plt.title('Moving Average of Collatz Stopping Times vs. Scaled Weierstrass Function')
    plt.xlabel('Starting number n')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()                      
    plt.tight_layout()
    plt.show()


# --- Plotting the numerical derivative ---
def plot_numerical_derivative(xs, ys, window=100):
    moving_avg = np.convolve(ys, np.ones(window)/window, mode='valid')
    
    # Calculate central difference.         
    # We lose one point at each end.
    # The x-values for the derivative will be xs[window-1:][1:-1]
    numerical_derivative = (moving_avg[2:] - moving_avg[:-2]) / 2.0 # (x_i+1 - x_i-1) = 2 for integer steps
    
    # Trim xs to match the derivative's length
    deriv_xs = xs[window-1:][1:-1]

    plt.figure(figsize=(14, 7))           
    plt.plot(deriv_xs, numerical_derivative, color='green', label=f'Numerical Derivative of Moving Average (window={window})')
									  
    # Add a horizontal line at y=0 for reference
    plt.axhline(0, color='grey', linestyle=':', linewidth=0.8)

    plt.title('Numerical Derivative of Collatz Moving Average')
    plt.xlabel('Starting number n')
    plt.ylabel('Approximate Derivative')                                                                   
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    


def plot_fft_analysis(moving_avg, window=200, sample_spacing=1):
    """
    Perform FFT on the moving average, plot the power spectrum, and estimate the power-law decay.
    """                             
    n = len(moving_avg)
    window = windows.blackman(len(moving_avg))
    windowed_data = (moving_avg - np.mean(moving_avg)) * window
    yf = fft(windowed_data)
    #yf = fft(moving_avg - np.mean(moving_avg))  # Remove DC offset
    xf = fftfreq(n, sample_spacing)[:n//2]      # Positive frequencies
    power = np.abs(yf[0:n//2]) ** 2             # Power spectrum

    # Plot
    plt.figure(figsize=(12, 6))
    plt.loglog(xf[1:], power[1:], color='blue', label='Power spectrum')  # Skip xf[0] (DC)
    
    # Fit power-law decay (exclude zero/negative power)
    mask = (xf[1:] > 0) & (power[1:] > 0)
    freqs_fit = xf[1:][mask]                                                                                    
    power_fit = power[1:][mask]
    
    if len(freqs_fit) > 1:
        coeffs = np.polyfit(np.log(freqs_fit), np.log(power_fit), 1)
        beta = -coeffs[0]        
        plt.plot(freqs_fit, np.exp(coeffs[1]) * freqs_fit**coeffs[0], 
                'r--', label=f'Fit: β = {beta:.2f}')                         
    
    plt.title(f'FFT of Collatz Moving Average (window={window})')
    plt.xlabel('Frequency (1/n)')
    plt.ylabel('Power')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"Power-law exponent (beta): {beta:.2f}")

if __name__ == "__main__":
    max_n = 50000                            
    xs, ys = compute_stopping_times(max_n)

    # plot_stopping_times(xs, ys)
    # plot_histogram(ys)

    # Plot the moving average with Weierstrass overlay
    plot_moving_average_with_weierstrass(xs, ys, window=200, weierstrass_a=0.6, weierstrass_b=5, weierstrass_terms=150)

    # Plot the numerical derivative
    plot_numerical_derivative(xs, ys, window=200)  # Use the same window for consistency

    # Compute the moving average and run FFT analysis
    window = 200
    moving_avg = np.convolve(ys, np.ones(window)/window, mode='valid')
    plot_fft_analysis(moving_avg, window=window, sample_spacing=1)
    
    
