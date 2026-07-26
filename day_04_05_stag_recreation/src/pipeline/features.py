import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
from src.pipeline.dataset import resample_signal

def extract_spectrogram(acc_z, fs_source=400, fs_target=500, n_bins=30, f_min=62.5, f_max=250.0):
    """
    Resamples accelerometer Z-axis data to fs_target (500 Hz),
    performs STFT, and extracts n_bins in the frequency range (f_min, f_max].
    Returns:
        spectrogram: np.array of shape (frames, n_bins)
    """
    # 1. Resample signal to target sampling rate (500 Hz)
    t_source = np.arange(len(acc_z)) * (1.0 / fs_source)
    t_target = np.arange(int(len(acc_z) * fs_target / fs_source)) * (1.0 / fs_target)
    
    # values shape is (1, N) for 1D resampling
    acc_z_resampled = resample_signal(t_source, acc_z.reshape(1, -1), t_target, kind='cubic')[0]
    
    # 2. STFT parameters: window = 80 ms (40 samples), hop = 20 ms (10 samples)
    nperseg = int(0.08 * fs_target) # 40 samples
    noverlap = nperseg - int(0.02 * fs_target) # 30 samples overlap -> 10 samples hop
    nfft = 128 # FFT points for higher frequency resolution
    
    f, t, Zxx = signal.stft(acc_z_resampled, fs=fs_target, window='hann', 
                             nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    # Magnitude spectrogram
    magnitude = np.abs(Zxx) # Shape: (nfft // 2 + 1, frames)
    
    # 3. Filter frequency range (f_min, f_max]
    idx_range = (f > f_min) & (f <= f_max)
    f_filtered = f[idx_range]
    magnitude_filtered = magnitude[idx_range, :] # Shape: (filtered_bins, frames)
    
    # 4. Interpolate along the frequency axis to get exactly n_bins (30)
    # We want to map f_filtered to a uniform grid of n_bins
    if magnitude_filtered.shape[0] < 2:
        # Fallback if signal is too short or frequency range is empty
        return np.zeros((magnitude.shape[1], n_bins))
        
    f_target_grid = np.linspace(f_filtered[0], f_filtered[-1], n_bins)
    
    spectrogram_frames = []
    for col in range(magnitude_filtered.shape[1]):
        interp_func = interpolate.interp1d(f_filtered, magnitude_filtered[:, col], kind='linear', fill_value="extrapolate")
        spectrogram_frames.append(interp_func(f_target_grid))
        
    # Stack frames and return shape (frames, n_bins)
    spectrogram = np.vstack(spectrogram_frames)
    
    # Apply log scaling to magnitude values to represent decibels (dB-like)
    spectrogram = np.log1p(spectrogram)
    
    return spectrogram
