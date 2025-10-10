# Import general
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory so custum imports work within this file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom libraries
import gpuRIR
from Lib.source_and_array_positions import fixed_position_from_angles, angles_from_array_reference, get_microphone_position
from Lib.signal_processing import normalize_audio
from Lib.utils import get_signal


def simulate_rir_1source(signal, source_pos,
                         fs, FRAME_SIZE_MS, HOP_RATIO, 
                         mic_pos, mic_center_pos, mic_rot_mat,
                         room_dims, T60=0.08, att_diff=60.0, att_max=60.0):
    """
    Simulate the measured audio signal of one fixed or moving source with a microphone array.

    Args:
        target_signal (np.ndarray): Target source signal.
        noise_signal (np.ndarray): Noise source signal.
        mic_pos (np.ndarray): Microphone positions.
        target_source_position (np.ndarray): Target source position(s).
        noise_source_position (np.ndarray): Noise source position(s).
        fs (int): Sample frequency.
        T60 (float): Reverberation time in seconds.
        att_diff (float): Attenuation when starting the diffuse reverberation model [dB].
        att_max (float): Attenuation at the end of the simulation [dB].
        ROOM_RADIUS (float): Radius of the room in meters.
        moving_target_source (bool): Whether the target source is moving.
        moving_noise_source (bool): Whether the noise source is moving.
        nb_of_rotations (int): Number of rotations for moving sources.

    Returns:
        noisy_signal (np.ndarray): Noisy signal with target and noise sources.
    """
    ## Number of samples, frames and channels
    frame_size = int(FRAME_SIZE_MS/1000*fs)
    hop_size = int(frame_size*HOP_RATIO)

    # Parameters for RIR calculation
    beta = gpuRIR.beta_SabineEstimation(room_dims, T60) # Reflection coefficients
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n( Tdiff, room_dims )	# Number of image sources in each dimension

    ## --- MICROPHONE ARRAY POSITION IN ROOM COORDINATES --- ##
    pos_rcv = get_microphone_position(mic_pos, mic_center_pos, mic_rot_mat)
    mic_pattern = "omni" # Receiver polar pattern  {"omni", "homni", "card", "hypcard", "subcard", "bidir"}

    ## --- FIXED SOURCE POSITION --- ##
    fixed_source_position = np.expand_dims(source_pos, axis=0)
    source_position = fixed_source_position

    ## --- RIR CALCULATION AND FILTERED SIGNAL --- ##
    # RIR Target source calculation
    RIRs_target = gpuRIR.simulateRIR(room_dims, beta, source_position, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, mic_pattern=mic_pattern)

    # Filter signal with RIRs to obtain mic signals
    filtered_signal = gpuRIR.simulateTrajectory(signal, RIRs_target)

    # Normalize (RMS) signal
    filtered_signal_norm = normalize_audio(filtered_signal, target_dbfs=-23)  # Normalize to -23 dBFS
    
    ## --- Output source DOAs at each position, with same size as the fft-ed signal --- ##
    # Total number of frames in the STFT of the measured signal
    mes_sig_nb_of_samples = filtered_signal_norm.shape[0]
    mes_sig_nb_of_frames = int((mes_sig_nb_of_samples - frame_size + hop_size) / hop_size)
    
    # source DOAs : All frames get the same DOA as the target source position
    source_doas = np.zeros((mes_sig_nb_of_frames,2), dtype=np.float32)
    source_doas[:,:] = angles_from_array_reference(source_pos, mic_center_pos, mic_rot_mat)
    
    return filtered_signal_norm, source_doas


if __name__ == "__main__":
    #############################################
    ########## PARAMETERS AND PATHS #############
    #############################################
    # Signal parameters
    FRAME_SIZE_MS = 32
    HOP_RATIO = 0.25
    
    # Room parameters
    '''
    Anechoic chamber : T60 = 0.08s
    Small classroom  : T60 = 0.65s
    Large classroom  : T60 = 1.39s
    '''
    room_dims = np.array([10.0, 10.0, 3.0])  # Room dimensions [x, y, z] in meters
    T60 = 0.08 # Time for the RIR to reach 60dB of attenuation [s]
    att_diff = 60.0	# Attenuation when start using the diffuse reverberation model [dB]
    att_max = 60.0 # Attenuation at the end of the simulation [dB]
    
    # Sources position
    source_pos = np.array([2.0, 3.0, 1.5])  # Source position in meters

    # Signal paths and name
    SIG_REL_PATH = '../Sound/Speech/'
    SIG_NAME = 'Harvard_list_01_1_16kHz'  
    IS_SIG_WHITE_NOISE = False
    MIC_ARRAY_NAME = 'half_sphere_array_pos_16mics'  # 
    MIC_POS_PATH = '../Mic_pos/for_comparison/3D/'+MIC_ARRAY_NAME

    #############################################
    ######### IMPORT SIGNALS AND FFT ############
    #############################################
    sig_path = os.path.realpath(os.path.join(os.path.dirname(__file__), SIG_REL_PATH, SIG_NAME+'.wav'))
    fs, sig, T = get_signal(sig_path)

    if IS_SIG_WHITE_NOISE:
        sig = np.random.normal(0, 1, len(sig))
        target_name = 'white_noise'

    # Array position
    mic_pos = np.load(os.path.realpath(os.path.join(os.path.dirname(__file__),MIC_POS_PATH+'.npy')))  # nMic x 3
    mic_center_pos = np.array([room_dims[0]/2, room_dims[1]/2, room_dims[2]/2])  # Center of the microphone array in the room
    mic_rot_mat = None  # No rotation matrix, use the default orientation of the microphone
    
    # Call function
    filtered_signal_norm, source_doas = simulate_rir_1source(
        signal=sig,
        source_pos=source_pos,
        fs=fs,
        FRAME_SIZE_MS=FRAME_SIZE_MS,
        HOP_RATIO=HOP_RATIO,
        mic_pos=mic_pos, 
        mic_center_pos = mic_center_pos,
        mic_rot_mat=mic_rot_mat,
        room_dims=room_dims,
        T60=T60, 
        att_diff=att_diff, 
        att_max=att_max
    )
    
    # Plot the first channel of the filtered signal
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(filtered_signal_norm)) / fs, filtered_signal_norm[:, 0], label='Filtered Signal', color='blue')
    plt.title('Filtered Signal (First Channel)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()