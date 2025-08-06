import random
import numpy as np
import torch
from scipy.signal import butter, filtfilt


def lowpass_filter(signal, cutoff=50, fs=500, order=4):
    is_torch = torch.is_tensor(signal)

    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered_signal = filtfilt(b, a, signal, axis=-1)

    return filtered_signal if not is_torch else torch.from_numpy(filtered_signal.copy()).float()

def highpass_filter(ecg, cutoff=0.5, fs=500, order=4):
    is_torch = torch.is_tensor(ecg)

    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    filtered_signal = filtfilt(b, a, ecg, axis=-1)

    return filtered_signal if not is_torch else torch.from_numpy(filtered_signal.copy()).float()


def add_white_gaussian_noise(ecg_signal, snr_db=(-6, 24), p=0.5, seed=None):
    """
    Add White Gaussian Noise (WGN) to the ECG signal.
    """
    np.random.seed(seed)
    random.seed(seed)

    is_torch = torch.is_tensor(ecg_signal)
    noisy_ecg = ecg_signal.clone().numpy() if is_torch else np.copy(ecg_signal)
    # Compute signal power
    signal_power = np.mean(noisy_ecg**2, axis=1, keepdims=True)
    # Compute noise power based on desired SNR (signal-to-noise ratio)
    if not isinstance(snr_db, (float, int)):
        snr_1, snr_2 = snr_db
        snr_db = random.uniform(snr_1, snr_2)

    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    # Generate WGN
    noise = np.random.normal(0, np.sqrt(noise_power), noisy_ecg.shape)

    # each lead has a probability of p to add noise
    mask = np.random.rand(noisy_ecg.shape[0]) >= p
    noise[mask] = 0.0  # add noise to some leads

    noisy_ecg += noise
    return noisy_ecg if not is_torch else torch.from_numpy(noisy_ecg)

def add_powerline_interference(ecg_signal, fs=500, amplitude=(0.05, 0.1), p=0.5, seed=None):
    np.random.seed(seed)
    random.seed(seed)

    is_torch = torch.is_tensor(ecg_signal)
    noisy_ecg = ecg_signal.clone().numpy() if is_torch else np.copy(ecg_signal)

    num_channels, seq_length = noisy_ecg.shape
    t = np.arange(seq_length) / fs

    if not isinstance(amplitude, (float, int)):
        amp_1, amp_2 = amplitude
        amplitude = random.uniform(amp_1, amp_2)

    frequency = np.random.choice([50, 60], 1)[0]
    power_noise = amplitude * np.sin(2 * np.pi * frequency * t)

    mask = np.random.rand(num_channels) < p
    power_noise = np.expand_dims(mask, 1) * np.expand_dims(power_noise, 0)  # add noise to some leads

    noisy_ecg += power_noise
    return noisy_ecg if not is_torch else torch.from_numpy(noisy_ecg)

def add_baseline_wander(ecg_signal, fs=500, freq=(0.1, 0.3), amplitude=(0.1, 0.5), p=0.5, seed=None):
    """
    Add Baseline Wander to the ECG signal.
    """
    np.random.seed(seed)
    random.seed(seed)

    is_torch = torch.is_tensor(ecg_signal)
    noisy_ecg = ecg_signal.clone().numpy() if is_torch else np.copy(ecg_signal)

    num_channels, seq_length = noisy_ecg.shape

    t = np.arange(seq_length) / fs

    if not isinstance(amplitude, (float, int)):
        amp_1, amp_2 = amplitude
        amplitude = random.uniform(amp_1, amp_2)

    if not isinstance(freq, (float, int)):
        f_1, f_2 = freq
        freq = random.uniform(f_1, f_2)

    baseline = amplitude * np.sin(2 * np.pi * freq * t)

    mask = np.random.rand(num_channels) < p
    baseline = np.expand_dims(mask, 1) * np.expand_dims(baseline, 0)  # add noise to some leads

    noisy_ecg += baseline
    return noisy_ecg if not is_torch else torch.from_numpy(noisy_ecg)

def add_emg_noise(ecg_signal, amplitude=(0.1, 0.5), burst_freq=(0.01, 0.2), p=0.5, seed=None):
    """
    Add Muscle (EMG) Noise to the ECG signal.
    """
    np.random.seed(seed)
    random.seed(seed)

    is_torch = torch.is_tensor(ecg_signal)
    noisy_ecg = ecg_signal.clone().numpy() if is_torch else np.copy(ecg_signal)

    num_channels, seq_length = noisy_ecg.shape

    if not isinstance(amplitude, (float, int)):
        amp_1, amp_2 = amplitude
        amplitude = random.uniform(amp_1, amp_2)

    if not isinstance(burst_freq, (float, int)):
        f_1, f_2 = burst_freq
        burst_freq = random.uniform(f_1, f_2)

    noise = amplitude * np.random.randn(*noisy_ecg.shape)

    mask = np.random.rand(num_channels) < p
    emg = np.zeros_like(noisy_ecg)
    for i in range(num_channels):
        # Introduce bursts randomly across time
        if not mask[i]:
            continue
        mask_emg = np.random.uniform(0, 1, seq_length) < burst_freq
        emg[i, :] = noise[i, :] * mask_emg

    noisy_ecg += emg
    return noisy_ecg if not is_torch else torch.from_numpy(noisy_ecg)


def add_predefined_noise(signal, noise, snr_db=(-6, 24), p=0.5, seed=None):
    """
    Adds predefined noise to a signal at a specified SNR.

    Args:
        signal (numpy.ndarray): The input clean signal (1D or 2D array).
        noise (numpy.ndarray): The predefined noise signal (same shape as the input signal).
        snr_db (float): Desired SNR in decibels.
 
    Returns:
        numpy.ndarray | torch.Tensor: Noisy signal.
    """
    random.seed(seed)

    if not isinstance(snr_db, (float, int)):
        snr_1, snr_2 = snr_db
        snr_db = random.uniform(snr_1, snr_2)

    is_torch = torch.is_tensor(signal)
    noisy_ecg = signal.clone().numpy() if is_torch else np.copy(signal)
    noise = noise.numpy() if torch.is_tensor(noise) else noise

    noisy_ecg = noisy_ecg.astype(np.float32)
    noise = noise.astype(np.float32)

    # Ensure the noise matches the shape of the signal
    if len(noise) != noisy_ecg.shape[-1]:
        raise ValueError("Noise and signal must have the same shape.")
    # Compute the power of the signal and noise
    signal_power = np.mean(noisy_ecg**2, axis=1, keepdims=True)
    noise_power = np.mean(noise**2)
    # Calculate the scaling factor for noise
    snr_linear = 10 ** (snr_db / 10)
    scaling_factor = np.sqrt(signal_power / (noise_power * snr_linear))

    mask = np.random.rand(noisy_ecg.shape[0]) < p

    # Scale the noise
    scaled_noise = noise * scaling_factor
    scaled_noise = scaled_noise * np.expand_dims(mask, -1)

    # Add scaled noise to the signal
    noisy_ecg = noisy_ecg + scaled_noise
    return noisy_ecg if not is_torch else torch.from_numpy(noisy_ecg)


def add_nst(signal, noise, snr_db=(-10, 0), p=0.5, seed=None):
    np.random.seed(seed)
    random.seed(seed)

    if not isinstance(snr_db, (float, int)):
        snr_1, snr_2 = snr_db
        snr_db = random.uniform(snr_1, snr_2)

    idx = np.random.choice(range(len(noise)))
    # Length of the signal
    signal_length = signal.shape[-1]

    is_torch = torch.is_tensor(signal)
    noisy_ecg = signal.clone().numpy() if is_torch else np.copy(signal)
    noise = noise.numpy() if torch.is_tensor(noise) else noise

    noisy_ecg = noisy_ecg.astype(np.float32)
    noise = noise.astype(np.float32)
    
    # Randomly select a segment of noise with the same length as the signal
    start_index = np.random.randint(0, noise.shape[-1] - signal_length)
    noise_segment = noise[idx, start_index:start_index + signal_length]
    
    # Calculate the power of the signal and noise
    signal_power = np.mean(noisy_ecg**2, axis=1, keepdims=True)
    noise_power = np.mean(noise_segment ** 2)
    
    # Calculate the desired noise power for the given SNR
    snr_linear = 10 ** (snr_db / 10)
    desired_noise_power = signal_power / snr_linear
    
    # Scale the noise to achieve the desired noise power
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    scaled_noise = noise_segment * scaling_factor
    
    # Choose a random channel to add noise
    mask = np.random.rand(noisy_ecg.shape[0]) < p
    # Add the scaled noise to the signal
    noisy_ecg = noisy_ecg + scaled_noise * np.expand_dims(mask, -1)
    
    return noisy_ecg if not is_torch else torch.from_numpy(noisy_ecg)


def crop_and_mask_noise(noise, target_channels=12, target_length=5000, p=0.5, seed=None):
    """
    noise: (C, Seq_len)
    """
    np.random.seed(seed)
    random.seed(seed)

    idx = np.random.choice(range(len(noise)))
    
    # Randomly select a segment of noise with the same length as the signal
    start_index = np.random.randint(0, noise.shape[-1] - target_length)
    noise_segment = noise[idx, start_index:start_index + target_length]
    
    # Choose a random channel to add noise
    mask = np.random.rand(target_channels) < p

    return noise_segment * np.expand_dims(mask, -1), ~mask     # 12, 5000


def add_combined_noise(signal, noise, snr_db=(-10, 0)):
    is_torch = torch.is_tensor(signal)
    if not isinstance(snr_db, (float, int)):
        snr_1, snr_2 = snr_db
        snr_db = random.uniform(snr_1, snr_2)

    noisy_ecg = signal.clone().numpy() if is_torch else np.copy(signal)
    noise = noise.numpy() if torch.is_tensor(noise) else noise

    noisy_ecg = noisy_ecg.astype(np.float32)
    noise = noise.astype(np.float32)

    # Calculate the power of the signal and noise
    signal_power = np.mean(noisy_ecg**2, axis=1, keepdims=True)
    noise_power = np.mean(noise ** 2, axis=1, keepdims=True)
    
    # Calculate the desired noise power for the given SNR
    snr_linear = 10 ** (snr_db / 10)
    desired_noise_power = signal_power / snr_linear
    
    # Scale the noise to achieve the desired noise power
    scaling_factor = np.sqrt(np.divide(desired_noise_power, noise_power, out=np.zeros_like(noise_power), where=noise_power != 0))
    scaled_noise = noise * scaling_factor   
    
    # Add the scaled noise to the signal
    noisy_ecg = noisy_ecg + scaled_noise
    
    return noisy_ecg if not is_torch else torch.from_numpy(noisy_ecg)