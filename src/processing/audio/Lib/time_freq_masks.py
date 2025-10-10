import numpy as np


def ideal_masks_from_target(X, V, Y, mask_on_reference = True, eps=1e-10):
    '''
    Compute different masks from oracle signals.
        
    Args:
        X (np.ndarray): 
            The spectrum of target signal [nb_of_channels, nb_of_frames, nb_of_bins]
        V (np.ndarray):
            The spectrum of noise signal [nb_of_channels, nb_of_frames, nb_of_bins]
        Y (np.ndarray):
            The spectrum of noisy signal [nb_of_channels, nb_of_frames, nb_of_bins]
        
        mask_on_reference (bool): If True, compute mask on first channel and broadcast to all channels.
                                  If False, compute mask for each channel independently.
        eps (float): Small constant to avoid division by zero.
    
    Returns:
        IBM (np.ndarray): 
            Ideal Binary Mask [nb_of_frames, nb_of_bins, nb_of_channels]
        IRM (np.ndarray): 
            Ideal Ratio Mask [nb_of_frames, nb_of_bins, nb_of_channels]
        PSM (np.ndarray): 
            Phase Sensitive Mask [nb_of_frames, nb_of_bins, nb_of_channels]
    '''
    X = np.moveaxis(X, 0, -1)  # [nb_of_frames, nb_of_bins, nb_of_channels]
    V = np.moveaxis(V, 0, -1)  # [nb_of_frames, nb_of_bins, nb_of_channels]
    Y = np.moveaxis(Y, 0, -1)  # [nb_of_frames, nb_of_bins, nb_of_channels]
    
    abs_X = np.abs(X)
    abs_V = np.abs(V)
    abs_Y = np.abs(Y)
    
    if mask_on_reference:
        # Compute masks on the first channel, then broadcast to all channels
        reference_channel = 0
        
        
        abs_X_ref = abs_X[..., reference_channel]
        abs_V_ref = abs_V[..., reference_channel]
        abs_Y_ref = abs_Y[..., reference_channel]
        X_ref = X[..., reference_channel]
        Y_ref = Y[..., reference_channel]

        # IBM
        IBM = (abs_X_ref > abs_V_ref).astype(np.float32)
        IBM = np.repeat(IBM[..., np.newaxis], abs_X.shape[-1], axis=-1)

        # IRM
        IRM = abs_X_ref / (abs_X_ref + abs_V_ref + eps)
        IRM = np.clip(IRM, 0, 1)
        IRM = np.repeat(IRM[..., np.newaxis], abs_X.shape[-1], axis=-1)

        # PSM
        phase_diff = np.angle(X_ref) - np.angle(Y_ref)
        PSM = abs_X_ref / (abs_Y_ref + eps) * np.cos(phase_diff)
        PSM = np.clip(PSM, -1, 1)
        PSM = np.repeat(PSM[..., np.newaxis], abs_X.shape[-1], axis=-1)
    
    else:
        # Compute masks for each channel independently
        IBM = (abs_X > abs_V).astype(np.float32)
        
        IRM = abs_X / (abs_X + abs_V + eps)
        IRM = np.clip(IRM, 0, 1)
        
        phase_diff = np.angle(X) - np.angle(Y)
        PSM = abs_X / (abs_Y + eps) * np.cos(phase_diff)
        PSM = np.clip(PSM, -1, 1)
        
    return IBM, IRM, PSM


def calculate_estimated_spectrum(Y, mask):
    '''
    Calculate the estimated spectrum from the noisy signal and the mask.
    
    Args:
        Y (np.ndarray): 
            The spectrum of noisy signal [nb_of_channels, nb_of_frames, nb_of_bins]
        mask (np.ndarray):
            The mask for target signal [nb_of_frames, nb_of_bins, nb_of_channels].
    
    Returns:
        X (np.ndarray):
            The estimated spectrum of target signal [nb_of_channels, nb_of_frames, nb_of_bins].
        V (np.ndarray):
            The estimated spectrum of noise signal [nb_of_channels, nb_of_frames, nb_of_bins].
    '''
    Y = np.moveaxis(Y, 0, -1)  # [nb_of_frames, nb_of_bins, nb_of_channels]
    X = Y * mask
    V = Y * (1 - mask)
    
    X = np.moveaxis(X, -1, 0)  # [nb_of_channels, nb_of_frames, nb_of_bins]
    V = np.moveaxis(V, -1, 0)  # [nb_of_channels, nb_of_frames, nb_of_bins]
    return X, V