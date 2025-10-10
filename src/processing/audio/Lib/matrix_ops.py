import numpy as np

def shrinkage_diagonal_loading(SCMs, shrinkage_intensity=0.01, eps=0):
    """
    Apply shrinkage-based diagonal loading.

    Args:
        SCMs (np.ndarray):
            The spatial covariance matrices to load for every frequency bins [nb_of_bins, nb_of_channels, nb_of_channels]
        shrinkage_intensity (float):
            Shrinkage intensity parameter.
    Returns:
        loaded_SCM (np.ndarray):
            The loaded spatial covariance matrix [nb_of_bins, nb_of_channels, nb_of_channels].
    """
    nb_of_bins, nb_of_channels, _  = SCMs.shape
    I = np.eye(nb_of_channels)

    # Initialize the array to hold the loaded covariance matrices
    loaded_SCM = np.zeros_like(SCMs)

    for bin_id in range(nb_of_bins):
        SCM = SCMs[bin_id, :, :]
        mean_variance = np.trace(SCM) / nb_of_channels
        target = mean_variance * I
        regularized_SCM = (1 - shrinkage_intensity) * SCM + shrinkage_intensity * target
        loaded_SCM[bin_id, :, :] = regularized_SCM + eps*I

    return loaded_SCM


def symmetrize_matrix(matrix):
    """
    Symmetrize a square matrix by averaging it with its hermitian.
    
    Args:
        matrix (np.ndarray): 
            The input square matrix to symmetrize [nb_of_bins, nb_of_channels, nb_of_channels].
    
    Returns:
        np.ndarray: 
            The symmetrized matrix [nb_of_bins, nb_of_channels, nb_of_channels].
    """
    return (matrix + np.conj(np.transpose(matrix, axes=(0, 2, 1)))) / 2.0


def compute_coherence_matrix_from_SCMs(SCM, eps=1e-10):
    """
    Compute complex coherence matrices from batched SCMs.
    
    Args:
        SCM (np.ndarray):
            Spatial covariance matrix [nb_frames, nb_bins, nb_channels, nb_channels].
        eps (float):
            Small constant to avoid division by zero.

    Returns:
        coherence_matrix (np.ndarray)
            Same shape as SCM.
    """
    # Extract diagonal for each frame and bin: shape (nb_frames, nb_bins, nb_channels)
    diag = np.diagonal(SCM, axis1=-2, axis2=-1)

    # Compute outer product of sqrt(diag) and its conjugate: shape (frames, bins, ch, ch)
    denom = np.sqrt(
        diag[..., :, None] * np.conj(diag[..., None, :])
    ) + eps

    # Element-wise division
    coherence_matrix = SCM / denom
    return coherence_matrix


def SCM_rolling_average(Phi, FIR_FILTER_LEN = 10, ALPHA_FIR = 1/20, ALPHA_IIR = 1/20):
    '''
    SCM_rolling_average applies a rolling average filter to the spatial covariance matrix (Phi).
    
    Args:
        Phi (np.ndarray): 
            The spatial covariance matrix [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        FIR_FILTER_LEN (int): 
            The length of the FIR filter for the rolling average.
        ALPHA_FIR (float): 
            The forgetting factor for the FIR filter.
        ALPHA_IIR (float): 
            The forgetting factor for the IIR filter.
    
    Returns:
        Phi_FIR (np.ndarray):
            The filtered spatial covariance matrix using FIR filter [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        Phi_IIR (np.ndarray):
            The filtered spatial covariance matrix using IIR filter [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    '''
    nb_of_frames = Phi.shape[0]

    # Rolling average on Phi
    Phi_FIR = np.zeros(Phi.shape, dtype=np.csingle)
    Phi_IIR = np.zeros(Phi.shape, dtype=np.csingle)
    for frame_id in range(nb_of_frames):

        if frame_id == 0:
            Phi_FIR[frame_id,...] = Phi[frame_id,...]
            Phi_IIR[frame_id,...] = Phi[frame_id,...]

        elif frame_id < FIR_FILTER_LEN:
            Phi_FIR[frame_id,...] = Phi[frame_id,...]
            Phi_IIR[frame_id,...] = (ALPHA_IIR*Phi[frame_id,...] 
                                                   + (1-ALPHA_IIR)*Phi_IIR[frame_id-1,...]  )

        else:
            Phi_FIR[frame_id,...] = (ALPHA_FIR*Phi[frame_id,...] 
                                                   + (1-ALPHA_FIR)
                                                   *np.mean(Phi[frame_id-FIR_FILTER_LEN:frame_id,...], axis=0)  )
            Phi_IIR[frame_id,...] = (ALPHA_IIR*Phi[frame_id,...] 
                                                   + (1-ALPHA_IIR)*Phi_IIR[frame_id-1,...]  )

    return Phi_FIR, Phi_IIR