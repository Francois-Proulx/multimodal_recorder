# Localise source
# - Import mic array position
# - Import signal (from measurements or with known position)
# - TDOA calculation (GCC family)
# - Source localisation (SRP - steared response)

# import general
import numpy as np
import matplotlib.pyplot as plt


# Import ssl functions
from src.processing.audio.Lib.utils import get_signal
from src.processing.audio.Lib.signal_processing import stft, normalize_audio
from src.processing.audio.Lib.scm import crossSpecDensity
from src.processing.audio.Lib.grid_and_tdoas import fibonacci_half_sphere, fibonacci_sphere, calculate_tdoa
from src.processing.audio.Lib.localisation_algos import DOAs_id_from_SRP, DOAs_coordinate_from_DOAs_id, DOAs_from_DOAs_coordinates, SRP_PHAT_trad

# import utils functions
from src.utils.io import project_file

# Import visualisation functions if needed
from src.processing.audio.Lib.visualize import plot_DOAs_single_source


def localise_audio(file_path, mic_path, save_path=None, FRAME_SIZE_MS = 32, HOP_RATIO = 0.25, nb_points=500, grid_type='fibonacci_half_sphere'):
    """
    Estimate DOAs (in mic coordinate frame).
    Returns:
        t: time vector
        DOAs_trad: np.ndarray [n_frames, 2] (theta, phi)
        DOAs_coord_trad: np.ndarray [n_frames, 3] (x, y, z unit sphere)
    """

    fs, y_n, _ = get_signal(file_path)
    y_n = normalize_audio(y_n, target_dbfs=-23)

    f, t, Y_n = stft(y_n.transpose(), fs, FRAME_SIZE_MS, HOP_RATIO)
    YYs = crossSpecDensity(Y_n)

    mic_pos = np.load(mic_path)
    if grid_type == 'fibonacci_sphere':
        scan_grid = fibonacci_sphere(nb_points)
    elif grid_type == 'fibonacci_half_sphere':
        scan_grid = fibonacci_half_sphere(nb_points)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")
    TDOAs_scan = calculate_tdoa(mic_pos, scan_grid)

    SRP_trad, _ = SRP_PHAT_trad(YYs, TDOAs_scan, f)
    DOAs_id_trad = DOAs_id_from_SRP(SRP_trad)
    DOAs_coord_trad = DOAs_coordinate_from_DOAs_id(DOAs_id_trad, scan_grid)
    DOAs_trad = DOAs_from_DOAs_coordinates(DOAs_coord_trad)

    return t, DOAs_trad, DOAs_coord_trad


if __name__ == "__main__":
    # Simple test run
    
    FILE_PATH = project_file("data", "raw", "multimodal", "session_20251006_153902", "audio", "audio.wav")
    MIC_PATH = project_file("src", "processing", "audio", "Mic_pos", "half_sphere_array_pos_16mics_clockwise.npy")
    t, DOAs_trad, DOAs_coord_trad = localise_audio(FILE_PATH, MIC_PATH, FRAME_SIZE_MS=32, HOP_RATIO=0.25, nb_points=500, grid_type='fibonacci_half_sphere')
    
    # Visualize results
    plot_DOAs_single_source(DOAs_trad, t, filename=None)
    plt.show()

