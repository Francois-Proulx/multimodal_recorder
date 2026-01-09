import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.io import project_file
from src.utils.visualize import plot_euler, plot_quat
from src.utils.referential_change import (
    interpolate_quaternions,
    quaternion_to_euler,
    apply_rotation,
)
from src.processing.multimodal.visualize_multimodal import (
    plot_doa_and_yaw,
    plot_doa_world,
)

# Video
from src.processing.video.main_video import process_apriltag

# Audio
from src.processing.audio.main_audio import localise_audio
from src.processing.audio.Lib.localisation_algos import DOAs_from_DOAs_coordinates
from src.processing.audio.Lib.visualize import plot_DOAs_single_source


def trim_audio_to_video(audio_fft_timestamps, t_video, doas, doas_coord):
    """
    Trim audio FFT frames to fit within video timestamps.

    Parameters
    ----------
    audio_fft_timestamps : np.ndarray
        Timestamps of the middle of FFT frames (in seconds)
    t_video : np.ndarray
        Video timestamps (in seconds)
    doas : np.ndarray
        DOAs (theta, phi) for each FFT frame
    doas_coord : np.ndarray
        DOA coordinates (x, y, z) for each FFT frame

    Returns
    -------
    trimmed_audio_frames, trimmed_doas, trimmed_doas_coord
    """
    hop_size = audio_fft_timestamps[1] - audio_fft_timestamps[0]
    audio_start, audio_end = audio_fft_timestamps[0], audio_fft_timestamps[-1]
    video_start, video_end = t_video[0], t_video[-1]

    # Trim start
    if audio_start < video_start:
        n_trim = min(
            int(np.ceil((video_start - audio_start) / hop_size)),
            len(audio_fft_timestamps),
        )
        print(f"Trimming {n_trim} audio frames at start")
        audio_fft_timestamps = audio_fft_timestamps[n_trim:]
        doas = doas[n_trim:]
        doas_coord = doas_coord[n_trim:]

    # Trim end
    if audio_end > video_end:
        n_trim = min(
            int(np.ceil((audio_end - video_end) / hop_size)), len(audio_fft_timestamps)
        )
        print(f"Trimming {n_trim} audio frames at end")
        audio_fft_timestamps = audio_fft_timestamps[:-n_trim]
        doas = doas[:-n_trim]
        doas_coord = doas_coord[:-n_trim]

    return audio_fft_timestamps, doas, doas_coord


def fuse_audio_video(
    audio_file,
    mic_pos_file,
    video_session,
    tag_family,
    tag_size,
    FRAME_SIZE_MS=32,
    HOP_RATIO=0.25,
    nb_points=500,
    grid_type="fibonacci_sphere",
):
    # --- 1. Localise in mic coordinates ---
    t_audio, DOAs_trad, DOAs_coord_trad = localise_audio(
        audio_file,
        mic_pos_file,
        FRAME_SIZE_MS=FRAME_SIZE_MS,
        HOP_RATIO=HOP_RATIO,
        nb_points=nb_points,
        grid_type=grid_type,
    )

    plot_DOAs_single_source(DOAs_trad, t_audio, filename=None)

    # --- 2. Estimate orientation from video ---
    quats_video, _, _, _, video_timestamps = process_apriltag(
        video_session, tag_family, tag_size
    )

    # --- 3. Get localisation timestamps ---
    AUDIO_TIMESTAMP_FILE = str(audio_file).replace("audio.wav", "audio_timestamps.csv")
    audio_timestamps = pd.read_csv(AUDIO_TIMESTAMP_FILE)["timestamp"].to_numpy()

    # --- 4. Create new timestamps for FFT frames ---
    hop_size = (FRAME_SIZE_MS / 1000) * HOP_RATIO
    n = len(t_audio)
    audio_fft_timestamps = np.linspace(
        audio_timestamps[0] + hop_size / 2,
        audio_timestamps[0] + hop_size / 2 + hop_size * (n - 1),
        n,
    )

    # --- 5 Trim audio so imu interpolation works ---
    audio_fft_timestamps, doas_trad, doas_coord = trim_audio_to_video(
        audio_fft_timestamps, video_timestamps, DOAs_trad, DOAs_coord_trad
    )
    t = audio_fft_timestamps - audio_fft_timestamps[0]

    # --- 6 Interpolate quaternions to audio timestamps ---
    quat_interp, R = interpolate_quaternions(
        quats_video, video_timestamps, audio_fft_timestamps
    )
    plot_quat(
        quats_video,
        quat_interp,
        t1=video_timestamps,
        t2=audio_fft_timestamps,
        title="Original and Interpolated Quaternions",
    )

    # --- 6.1 (VALIDATION) Convert to Euler ---
    roll, pitch, yaw = quaternion_to_euler(
        quat_interp[:, 0], quat_interp[:, 1], quat_interp[:, 2], quat_interp[:, 3]
    )
    plot_euler(roll, pitch, yaw)
    plot_doa_and_yaw(
        t, doas_trad, sensors=[yaw], ids=["apriltag yaw"], title="SSL and apriltag yaw"
    )

    # # --- 6.2 (VALIDATION) Simulate rotation of a point ---
    # pB = np.tile(np.array([[1], [0], [0]]).reshape(3,), (len(quat_interp), 1))  # Point in sensor frame
    # pA = apply_rotation(R, pB)
    # pB_doas = DOAs_from_DOAs_coordinates(pB)
    # pA_doas = DOAs_from_DOAs_coordinates(pA)
    # plot_DOAs_single_source(pA_doas, t, filename=None)
    # plot_doa_alignement_and_yaw(t, pB_doas, yaw, pA_doas, title="Point [1,0,0] rotated by IMU orientation")

    # --- 7. Rotate DOAs into world frame ---
    DOAs_coord_world = apply_rotation(np.linalg.inv(R), doas_coord)
    DOAs_trad_world = DOAs_from_DOAs_coordinates(
        DOAs_coord_world, center_around_zero=True
    )
    plot_DOAs_single_source(DOAs_trad_world, t, filename=None)
    plot_doa_world(
        t,
        DOAs_trad_world,
        sensors=[yaw],
        ids=["apriltag yaw"],
        title="DOAs rotated into world frame my apriltag",
    )
    plt.show()

    return DOAs_trad_world, DOAs_coord_world


if __name__ == "__main__":
    # Paths to files
    AUDIO_FILE = project_file(
        "data", "raw", "multimodal", "session_20251015_125445", "audio", "audio.wav"
    )
    MIC_PATH = project_file(
        "src",
        "processing",
        "audio",
        "Mic_pos",
        "half_sphere_array_pos_16mics_clockwise.npy",
    )
    VIDEO_SESSION = session_path = project_file(
        "data", "raw", "multimodal", "session_20251015_125445"
    )

    # Parameters
    FRAME_SIZE_MS = 32
    HOP_RATIO = 0.5
    nb_points = 500
    grid_type = "fibonacci_half_sphere"
    tag_family = "tag25h9"
    tag_size = 0.140

    fuse_audio_video(
        AUDIO_FILE,
        MIC_PATH,
        VIDEO_SESSION,
        tag_family,
        tag_size,
        FRAME_SIZE_MS,
        HOP_RATIO,
        nb_points,
        grid_type,
    )
