import numpy as np
from numpy.fft import fft, ifft, fftfreq

def low_pass_filter(x, x0, order):
    """Low pass filter of arbitrary order, order=1 implies 1/f noise.
    
    Args:
        x: Frequency values
        x0: Cutoff frequency
        order: Filter order
        
    Returns:
        Filter response normalized to peak of 1
    """
    return 1./np.sqrt(1. + np.power(np.abs(x/x0), order))

def all_pass_filter(x, x0, order):
    """White noise filter (all frequencies pass through).
    
    Args:
        x: Frequency values
        x0: Unused parameter (kept for API consistency)
        order: Unused parameter (kept for API consistency)
        
    Returns:
        Constant value of 1
    """
    return 1

def decay_cos(t, A, tau, freq_rads):
    """Decaying cosine function.
    
    Args:
        t: Time values
        A: Amplitude
        tau: Decay time constant
        freq_rads: Frequency in radians
        
    Returns:
        Decaying cosine values
    """
    return A * np.exp(-t/tau) * np.cos(freq_rads*t)

def decay_cos_envelope(t, A, tau, freq_rads):
    """Envelope of decaying cosine function.
    
    Args:
        t: Time values
        A: Amplitude
        tau: Decay time constant
        freq_rads: Frequency in radians (unused)
        
    Returns:
        Exponential decay envelope
    """
    return A * np.exp(-t/tau)

def generate_noise_trajectory(sample_rate=1, t_max=200, relative_PSD_strength=(1e-3)**2, f0=1e-2, white=False):
    """Generate a single noisy trajectory for dephased system's frequency.
    
    Args:
        sample_rate: Sampling rate in us^-1
        t_max: Maximum time in us
        relative_PSD_strength: Relative power spectral density strength in (us^-2)/sample_rate
        f0: Cutoff frequency in us^-1
        white: If True, generates white noise. If False, generates 1/f filtered noise
        
    Returns:
        tuple: (time_list, filtered_freq_shifts, filtered_trajectory, freq_x_vals, freq_psd_vals)
            - time_list: List of time points
            - filtered_freq_shifts: Frequency shifts at each time point
            - filtered_trajectory: Cumulative phase shift vs time
            - freq_x_vals: Frequency values for PSD
            - freq_psd_vals: Power spectral density values
    """
    N = int(sample_rate * t_max) + 1
    t_list = np.linspace(0, t_max, N)
    dt = np.mean(np.diff(t_list))
    
    # Generate unfiltered time domain trajectory
    freq_shifts = np.sqrt(relative_PSD_strength * sample_rate) * np.random.normal(0., 1., N)
    
    # Convert to frequency domain
    freq_y_vals = fft(freq_shifts)
    freq_x_vals = fftfreq(len(freq_shifts), d=dt)
    
    # Apply appropriate filter
    freq_filter = all_pass_filter if white else low_pass_filter
    freq_y_vals = np.multiply(freq_y_vals, freq_filter(freq_x_vals, f0, order=1.0))
    
    # Convert back to time domain
    filtered_freq_shifts = np.real(ifft(freq_y_vals))
    filtered_trajectory = np.cumsum(filtered_freq_shifts) * dt
    
    # Calculate PSD
    freq_y_vals = fft(filtered_freq_shifts)
    freq_psd_vals = np.abs(freq_y_vals[:N//2])**2/sample_rate**2
    
    return t_list, filtered_freq_shifts, filtered_trajectory, freq_x_vals[:N//2], freq_psd_vals 

# Generate a bunch of trajectories and plot frequency spread and noise power spectral density
def monte_carlo_noise(num_realizations=500, sample_rate=1, t_max=200, relative_PSD_strength=(1e-3)**2, f0=1e-2, white=False):
    N = int(sample_rate*t_max) + 1                         # Total number of time points
    trajectories_list = np.zeros((num_realizations, N))    # List of trajectories
    noise_psd_list    = np.zeros((num_realizations, N//2)) # 'x//y' = int(x/y)
    
    # Simple for-loop
    trajectories_list, noise_psd_list = np.zeros((num_realizations, N)), np.zeros((num_realizations, N//2))
    for idx in range(num_realizations):
        t_list, _, traj, freq_list, psd = generate_noise_trajectory(sample_rate, t_max, relative_PSD_strength, f0, white)
        trajectories_list[idx] = traj
        noise_psd_list[idx] = psd
    
    return t_list, trajectories_list, freq_list, noise_psd_list