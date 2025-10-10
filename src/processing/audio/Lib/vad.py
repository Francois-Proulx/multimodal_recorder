'''
Voice Activity Detection (VAD) Functions
'''
import numpy as np
import torch
# import webrtcvad  # Uncomment if using webrtcvad

def get_silero_vad_mask(signal, fs):
    if not hasattr(get_silero_vad_mask, "model"):
        print("Loading Silero VAD model...")
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        get_silero_vad_mask.model = model
        get_silero_vad_mask.get_speech_timestamps = utils[0]  # get_speech_timestamps

    model = get_silero_vad_mask.model
    get_speech_timestamps = get_silero_vad_mask.get_speech_timestamps

    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).float()

    timestamps = get_speech_timestamps(signal, model, sampling_rate=fs)
    mask = np.zeros(signal.shape[0], dtype=bool)
    for ts in timestamps:
        mask[ts['start']:ts['end']] = True

    return mask


def sample_mask_to_frame_mask(sample_mask, fs, frame_size_ms=30, hop_ratio=0.25):
    """
    Convert a sample-wise boolean mask (e.g., from Silero) to a frame-wise boolean mask.

    Args:
        sample_mask (np.ndarray): Boolean array of shape [nb_samples] (True = speech)
        frame_length (int): Frame length in samples
        hop_length (int): Hop length in samples

    Returns:
        np.ndarray: Boolean mask over frames (True = speech in that frame)
    """
    frame_size = int(frame_size_ms/1000*fs)
    hop_size = int(frame_size*hop_ratio)
    
    num_samples = len(sample_mask)
    num_frames = 1 + int(np.floor((num_samples - frame_size) / hop_size))
    frame_mask = np.zeros(num_frames, dtype=bool)

    for frame_id in range(num_frames):
        start = frame_id * hop_size
        end = start + frame_size
        if np.any(sample_mask[start:end]):
            frame_mask[frame_id] = True

    return frame_mask


# def detect_speech_in_frames(frames, sample_rate):
#     """
#     Process each frame to detect speech presence.

#     Args:
#         frames (iterable): An iterable of audio frames.
#         sample_rate (int): The sample rate of the audio.

#     Returns:
#         list: A list of binary values (1 for speech, 0 for non-speech).
#     """
#     # Initialize the VAD
#     vad = webrtcvad.Vad()
#     vad.set_mode(3)  # Set aggressiveness mode (0 to 3)

#     speech_flags = []
#     for frame in frames:
#         # Ensure the frame is in bytes format
#         if isinstance(frame, bytes):
#             is_speech = vad.is_speech(frame, sample_rate)
#             speech_flags.append(1 if is_speech else 0)
#         else:
#             raise ValueError("Each frame must be in bytes format.")
#     return np.asarray(speech_flags, dtype=np.float32)