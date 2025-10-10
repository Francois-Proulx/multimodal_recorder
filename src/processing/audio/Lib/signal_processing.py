'''
Signal Processing Functions
Includes functions for normalisation, STFT, iSTFT, frame generation, and resampling.
'''

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


################## Normalisation Functions ###################
def normalize_audio(signal, speech_mask=None, target_dbfs=None, gain=None, return_gain=False):
    """
    Normalize an audio signal, optionally using only speech frames.

    Args:
        signal (np.ndarray): Audio signal, shape [samples] or [samples, channels].
        speech_mask (np.ndarray, optional): Boolean mask for speech regions (1D, length = samples).
                                             If None, use entire signal.
        target_dbfs (float, optional): Target level in dBFS. If None, normalize to 0 dB RMS. Usually -23 dBFS.
        gain (float, optional): Additional gain in dB after normalization.
        return_gain (bool, optional): If True, also return the gain factor used.

    Returns:
        np.ndarray or (np.ndarray, float): Normalized signal, and optionally the applied gain factor.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("Signal must be a NumPy array.")
    signal = signal.astype(np.float32)

    # Ensure 2D shape [samples, channels]
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    # Select region for RMS calculation
    if speech_mask is not None:
        if not (isinstance(speech_mask, np.ndarray) and speech_mask.dtype == bool):
            raise ValueError("Speech mask must be a NumPy boolean array.")
        if len(speech_mask) != signal.shape[0]:
            raise ValueError("Speech mask length must match signal length.")
        ref_channel = signal[:, 0]
        speech_only = ref_channel[speech_mask]
        if len(speech_only) == 0:
            print("Warning: No speech detected, skipping normalization.")
            return (signal, 1.0) if return_gain else signal
        rms_source = speech_only
    else:
        rms_source = signal

    # Compute RMS
    current_rms = np.sqrt((np.abs(rms_source) ** 2).mean())

    # Check if empty or silent signal - skip normalization
    if current_rms == 0:
        return signal

    # Compute target RMS
    if target_dbfs is None:
        target_rms = 1.0  # 0 dB RMS
    else:
        target_rms = 10 ** (target_dbfs / 20.0) # Convert dBFS to linear RMS

    gain_factor = target_rms / current_rms
    signal = signal * gain_factor

    # Apply extra gain
    if gain is not None:
        extra_gain = 10 ** (gain / 20.0)
        signal = signal * extra_gain
        gain_factor *= extra_gain

    # Sqeeze output if single channel
    signal = np.squeeze(signal)
    return (signal, gain_factor) if return_gain else signal


def peak_normalize(x):
    """
        Peak normalization, the maximum now becomes 1 or -1

        Args:
            x (Tensor): Data to normalize
    """
    factor = 1/(np.max(np.abs(x)))
    new_x = factor * x

    return new_x, factor


def verify_speech_rms(signal, fs, speech_mask=None, plot=False):
    """
    Compute the RMS level of speech regions in an audio signal (in dB relative to RMS=1).

    Args:
        signal (np.ndarray): Audio signal, shape [samples] or [samples, channels].
        fs (int): Sampling rate in Hz.
        speech_mask (np.ndarray, optional): Boolean mask for speech regions (1D, length = samples).
                                             If None, use entire signal.
        plot (bool, optional): Whether to plot the waveform with highlighted speech regions.

    Returns:
        float: RMS of the speech regions in dB relative to RMS=1.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("Signal must be a NumPy array.")
    signal = signal.astype(np.float32)

    # Ensure 2D shape [samples, channels]
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    # Select region for RMS calculation
    if speech_mask is not None:
        if not (isinstance(speech_mask, np.ndarray) and speech_mask.dtype == bool):
            raise ValueError("Speech mask must be a NumPy boolean array.")
        if len(speech_mask) != signal.shape[0]:
            raise ValueError("Speech mask length must match signal length.")
        ref_channel = signal[:, 0]
        speech_only = ref_channel[speech_mask]
        if len(speech_only) == 0:
            print("⚠️ No speech detected.")
            return None
    else:
        speech_only = signal[:, 0]

    # Compute RMS
    rms = np.sqrt((np.abs(speech_only) ** 2).mean())
    db_rms = 20 * np.log10(rms + 1e-12)  # Avoid log(0)

    print(f"📏 Speech RMS: {db_rms:.2f} dB")

    # Optional plotting
    if plot:
        time = np.arange(signal.shape[0]) / fs
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal[:, 0], label='Signal', alpha=0.6)
        if speech_mask is not None:
            plt.plot(time[speech_mask], signal[speech_mask, 0], color='r', lw=1, label='Speech Region')
        plt.title(f"Speech RMS: {db_rms:.2f} dB")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return db_rms


################### STFT Functions ###################
def stft(xs, fs, frame_size_ms=32, hop_ratio=0.25):
    """
    Perform STFT (not normalized!!!!)

    Args:
        xs (np.ndarray):
            Signals in the time domain [nb_of_channels, nb_of_samples].
        fs (int):
            Sample frequency.
        hop_size (int):
            Space in samples between windows.
        frame_size (int):
            Frams size in ms.
    Returns:
        Ys (np.ndarray):
            The time-frequency representation [nb_of_channels, nb_of_frames, nb_of_bins].
    """

    if np.ndim(xs) == 1:
        xs = np.expand_dims(xs, axis=0)

    frame_size = int(frame_size_ms/1000*fs)
    hop_size = int(frame_size*hop_ratio)

    nb_of_channels = xs.shape[0]
    nb_of_samples = xs.shape[1]
    nb_of_frames = int((nb_of_samples - frame_size + hop_size) / hop_size)
    nb_of_bins = int(frame_size/2+1)

    ws = np.tile(np.hanning(frame_size), (nb_of_channels, 1))
    Xs = np.zeros((nb_of_channels, nb_of_frames, nb_of_bins), dtype=np.csingle)

    for i in range(0, nb_of_frames):
        Xs[:, i, :] = np.fft.rfft(xs[:, (i*hop_size):(i*hop_size+frame_size)] * ws)

    f = np.linspace(0, fs//2, nb_of_bins)
    t = np.linspace(0, nb_of_samples/fs, nb_of_frames)

    return f, t, Xs


def istft(Xs, hop_ratio):
    """
    Perform iSTFT

    Args:
        Xs (np.ndarray):
            Signals in the frequency domain (nb_of_channels, nb_of_frames, nb_of_bins).
        hop_size (int):
            Space in samples between windows.
    Returns:
        (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_samples).
    """

    if np.ndim(Xs) == 2:
        Xs = np.expand_dims(Xs, 0)

    nb_of_channels = Xs.shape[0]
    nb_of_frames = Xs.shape[1]
    nb_of_bins = Xs.shape[2]
    
    frame_size = (nb_of_bins-1)*2
    hop_size = int(frame_size*hop_ratio)
    nb_of_samples = nb_of_frames * hop_size + frame_size - hop_size

    ws = np.tile(np.hanning(frame_size), (nb_of_channels, 1))
    xs = np.zeros((nb_of_channels, nb_of_samples), dtype=np.float32)

    for i in range(0, nb_of_frames):

        sample_start = i * hop_size
        sample_stop = i * hop_size + frame_size

        xs[:, sample_start:sample_stop] += np.fft.irfft(Xs[:, i, :]) * ws

    return xs

#################### Frame Generation Functions ###################
def frame_generator(signal, frame_size_ms, hop_ratio, fs):
    """
    Generate overlapping frames from a mono audio signal and convert each frame to 16-bit PCM bytes.

    Args:
        signal (np.ndarray): Mono audio signal with shape [nb_of_samples].
        frame_size_ms (float): Frame size in milliseconds.
        hop_ratio (float): Hop size as a fraction of the frame size.
        fs (int): Sampling frequency of the audio signal.

    Yields:
        bytes: A frame of audio data in 16-bit PCM format.
    """
    # Ensure the signal is a 1D array
    if signal.ndim != 1:
        raise ValueError("Input signal must be mono (1D array).")

    # Calculate frame size and hop size in samples
    frame_size = int(frame_size_ms / 1000 * fs)
    hop_size = int(frame_size * hop_ratio)
    nb_of_samples = len(signal)

    # Iterate over the signal to extract frames
    for start in range(0, nb_of_samples - frame_size + 1, hop_size):
        frame = signal[start:start + frame_size]
        # Convert the frame from float32 to 16-bit PCM bytes
        frame_pcm16 = float32_to_pcm16(frame)
        yield frame_pcm16

def float32_to_pcm16(audio_float32):
    """
    Convert a float32 numpy array to 16-bit PCM bytes.

    Args:
        audio_float32 (np.ndarray): Audio data in float32 format.

    Returns:
        bytes: Audio data in 16-bit PCM format.
    """
    # Ensure the audio is within the range [-1.0, 1.0]
    audio_float32 = np.clip(audio_float32, -1.0, 1.0)
    # Scale to 16-bit integer range and convert to int16
    audio_int16 = (audio_float32 * 32768).astype(np.int16)
    # Convert to bytes
    return audio_int16.tobytes()



##################### Resampling Functions ###################
def resample_signal(input_signal, original_fs, target_fs, cutoff_freq,  preserve_phase=True):
    """
    Resample a signal to a target sampling rate while applying a low-pass filter.
    An option is provided to use zero-phase filtering (to preserve phase) or
    standard causal filtering (which introduces a phase delay).

    ***Eventually would be interesting to design the lowpass filter that is used in resample_poly.***
    
    Args:
        input_signal (np.ndarray):
            The input time-domain signal (nb_of_samples, nb_of_channels).
        original_fs (float):
            Original sampling rate in Hz.
        target_fs (float):
            Target sampling rate in Hz.
        cutoff_freq (float):
            cutoff frequency of the low-pass filter in Hz.
        preserve_phase (bool):
            If True, use zero-phase filtering (sosfiltfilt) to avoid modifying the phase; otherwise, use causal filtering (sosfilt).
    Returns:
        resampled_signal_filtered (np.ndarray):
            The resampled time-domain signal using the signal.resample_poly function AND a low-pass filter before (nb_of_samples, nb_of_channels).
        resampled_signal_unfiltered (np.ndarray):
            The resampled time-domain signal using only the signal.resample_poly function (nb_of_samples, nb_of_channels).
        sos (np.ndarray):
            Low pass filter (8th-order Butterworth) used for resampled_signal_2_lowpass (4,6).
    """
    # Step 1: Design a low-pass filter (8th-order Butterworth)
    nyquist_rate = original_fs / 2.0
    normalized_cutoff = cutoff_freq / nyquist_rate
    sos = signal.butter(8, normalized_cutoff, btype='low', output='sos')

    # Step 2: Apply the low-pass filter
    if preserve_phase:
        # Zero-phase filtering (non-causal but preserves phase)
        filtered_signal = signal.sosfiltfilt(sos, input_signal, axis=0)
    else:
        # Standard causal filtering (introduces phase delay)
        filtered_signal = signal.sosfilt(sos, input_signal, axis=0)
    
    # Step 3: Resample the signal
    resampled_signal_filtered = signal.resample_poly(filtered_signal, target_fs, original_fs)
    resampled_signal_unfiltered = signal.resample_poly(input_signal, target_fs, original_fs)
    
    return resampled_signal_filtered, resampled_signal_unfiltered, sos
