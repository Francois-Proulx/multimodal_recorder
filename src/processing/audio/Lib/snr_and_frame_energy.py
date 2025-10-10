import numpy as np
import matplotlib.pyplot as plt


####################### SNR Computation #######################
def compute_snr(clean_signal, noise, eps=1e-12):
    """
    Compute SNR in decibels.

    Args:
        clean_signal (np.ndarray): Clean signal (target).
        noise (np.ndarray): Noise signal.
        eps (float): Small constant to avoid log(0).

    Returns:
        float: Estimated SNR in dB
    """
    # Compute power
    clean_signal_power = max(np.mean(np.abs(clean_signal) ** 2), eps)  # Avoid zero power; SNR = 0 dB if both signals are zero
    noise_power = max(np.mean(np.abs(noise) ** 2), eps)  # Avoid division by zero
    
    # Compute SNR
    snr = 10 * np.log10(clean_signal_power / noise_power)
    return snr


def compute_snr_stft(Xs_clean, Xs_noise, eps=1e-12):
    """
    Compute frame-wise SNR from STFTs of clean and noisy signals.

    Args:
        Xs_clean (np.ndarray): STFT of the clean signal, shape [channels, frames, bins].
        Xs_noise (np.ndarray): STFT of the noise signal, same shape as Xs_clean.
        eps (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: SNR values in dB for each frame, shape [frames].
    """
    # Compute power spectra
    power_clean = np.sum(np.abs(Xs_clean) ** 2, axis=(0, 2))  # Sum over channels and bins
    power_noise = np.sum(np.abs(Xs_noise) ** 2, axis=(0, 2))  # Sum over channels and bins
    
    # Ensure no zero power to avoid log(0)
    power_clean = np.maximum(power_clean, eps)  # Avoid zero power; SNR = 0 dB if both signals are zero
    power_noise = np.maximum(power_noise, eps)  # Avoid division by zero

    # Compute SNR in dB for each frame
    snr_db = 10 * np.log10(power_clean / power_noise)

    return snr_db


######################## Get top energy frames ########################
def compute_frame_energy(Xs):
    """
    Compute energy per frame from STFT.

    Args:
        Xs (np.ndarray): STFT of shape [nb_channels, nb_frames, nb_bins]

    Returns:
        np.ndarray: Power per frame [nb_frames]
    """
    power = np.sum(np.abs(Xs) ** 2, axis=(0, 2))  # Sum over freq bins and channels
    return power

def get_top_energy_frames(power, top_percent=0.1, plot=True):
    """
    Get average energy of the top X% frames.

    Args:
        power (np.ndarray): Power per frame [nb_frames]
        top_percent (float): Fraction of highest-energy frames to consider (e.g. 0.1 for top 10%)
        plot (bool): Whether to plot the energy and threshold

    Returns:
        float: Mean energy of top frames
        np.ndarray: Boolean mask of selected top frames
    """
    nb_frames = len(power)
    top_k = max(1, int(nb_frames * top_percent))
    
    # Sort to get the threshold
    sorted_power = np.sort(power)
    threshold = sorted_power[-top_k]
    
    # Mask of frames above or equal to threshold
    mask = power >= threshold
    top_mean_energy = np.mean(power[mask])
    
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(power, label="Frame Energy")
        plt.axhline(y=threshold, color='r', linestyle='--', label=f"Top {int(top_percent * 100)}% threshold")
        plt.fill_between(np.arange(len(power)), 0, power, where=mask, color='orange', alpha=0.3)
        plt.title("Frame Energy with Top Frame Highlight")
        plt.xlabel("Frame")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return top_mean_energy, mask


