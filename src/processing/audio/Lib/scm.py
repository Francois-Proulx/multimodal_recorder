import numpy as np
from src.processing.audio.Lib.matrix_ops import shrinkage_diagonal_loading, symmetrize_matrix

def crossSpecDensity(Ys, mask=None, eps=1e-10):
    """
    Perform Cross spectral density (or cross spectrum), which is the fourier transform of the cross correlation
    
    Posibility to us a time-frequency mask.

    Args:
        Ys (np.ndarray):
            The time-frequency representation [nb_of_channels, nb_of_frames, nb_of_bins]
        mask (np.ndarray):
            Mask for target source [nb_of_frames, nb_of_bins]
    Returns:
        YYs (np.ndarray):
            The cross spectrum [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
    """
    Ys = np.moveaxis(Ys, 0, -1)  # [nb_of_frames, nb_of_bins, nb_of_channels]
    if mask is not None:
        # Apply mask
        mask_expanded = mask[:,:,np.newaxis]
        Ys_masked = Ys * mask_expanded
        
        # Compute cross spectrum
        YYs = np.einsum('tfc,tfd->tfcd', Ys_masked, np.conj(Ys_masked))
        
        # Normalize by mask
        norm = mask_expanded[:,:,:,np.newaxis] + eps
        # YYs /= norm

    else:
        YYs = np.einsum('tfc,tfd->tfcd', Ys, np.conj(Ys))
        
    return YYs


#####################################################
############ Calculations offline SCM  ##############
#####################################################

def compute_offline_SCM_from_cross_spec(YYs):
    """
    Compute the Spatial Covariance Matrix as an average of the cross spectrum over all frames.

    Args:
        YYs (np.ndarray):
            The cross spectrum [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
    Returns:
        Phi_yy (np.ndarray):
            The spatial covariance matrix (SCM) [nb_of_bins, nb_of_channels, nb_of_channels]
    """
    Phi_yy = np.mean(YYs, axis=0)

    return Phi_yy

def compute_offline_SCM_from_spec(Ys):
    """
    Compute cross spectrum and SCM as an average of the cross spectrum.

    Args:
        Ys (np.ndarray):
            The time-frequency representation [nb_of_channels, nb_of_frames, nb_of_bins]
    Returns:
        Phi_yy (np.ndarray):
            The spatial covariance matrix (SCM) [nb_of_bins, nb_of_channels, nb_of_channels]
    """
    Ys = np.moveaxis(Ys, 0, -1)  # [nb_of_frames, nb_of_bins, nb_of_channels]
    YYs = np.einsum('tfc,tfd->tfcd', Ys, np.conj(Ys)) # [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
    Phi_yy = np.mean(YYs, axis=0)  # [nb_of_bins, nb_of_channels, nb_of_channels]

    return Phi_yy


#####################################################
######## Calculations of SCM at each frame ##########
#####################################################
def compute_real_time_SCMs_FIR(XX, nb_of_frames_for_average):
    '''
    Calculate the SCMs for each frame by averaging over a number of frames.
    
    Args:
        XX (np.ndarray): 
            The cross spectrum of target signal [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        nb_of_frames_for_average (int): 
            The number of frames to average over.
    
    Returns:
        Phi_XX (np.ndarray):
            The SCM at each frame [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    '''
    nb_of_frames, nb_of_bins, nb_of_channels, _ = XX.shape
    
    Phi_XX = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)
    
    for frame_id in range(nb_of_frames):
        if frame_id < nb_of_frames_for_average-1:
            Phi_XX[frame_id,...] = np.mean(XX[:frame_id+1,...], axis=0)
        else:
            Phi_XX[frame_id,...] = np.mean(XX[frame_id-nb_of_frames_for_average+1:frame_id+1,...], axis=0)
    
    return Phi_XX


def compute_real_time_SCMs_IIR(YYs, Alpha=0.5, normalize='none'):
    """
    Compute the Spatial Correlation Matrix at each frame with a rolling exponential average.

    Phi_yy[l] = alpha * YYs[l] + (1 - alpha) * Phi_yy[l-1]

    Works better if signals are stationary.

    Args:
        YYs (np.ndarray):
            The cross spectrum [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
        mask (np.ndarray):
            - Mono-channel mask [nb_of_frames, nb_of_bins]
            - Multi-channel mask [nb_of_frames, nb_of_bins, nb_of_channels]
        Alpha (float):
            Forgetting factor, 0 < alpha < 1. If alpha = 1, it "forgets" the last estimation and only looks at current observation.
    Returns:
        Phi_yy (np.ndarray):
            The spatial covariance matrix (SCM)  [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
    """
    nb_of_frames, nb_of_bins, nb_of_channels, _ = YYs.shape
    Phi_yy = np.zeros_like(YYs)

    for frame_id in range(nb_of_frames):
        if frame_id < nb_of_channels: # use average in first frames
            Phi_yy[frame_id] = np.mean(YYs[:frame_id+1], axis=0)
        else:
            Phi_yy[frame_id] = Alpha * YYs[frame_id] + (1 - Alpha) * Phi_yy[frame_id - 1]
            
        if normalize == 'amplitude':
            Phi_yy[frame_id] /= (np.abs(Phi_yy[frame_id]) + 1e-10)  # Avoid division by zero
        elif normalize == 'trace':
            for bin_id in range(nb_of_bins):
                trace = np.trace(Phi_yy[frame_id, bin_id])
                if trace != 0:
                    Phi_yy[frame_id, bin_id] /= trace
        elif normalize == 'none':
            pass

    return Phi_yy



def compute_theoretical_SCMs_from_TDOAs(f, taus, TDOAs_scan, fs):
    """
    Calculate theoretical Spatial Covariance Matrix in real time from TDOAs

    Args:
        f (np.ndarray):
            Frequency bins [nb_of_bins, ]
        taus (np.ndarray):
            The maximum value of the cross correlation for each frame [nb_of_channels,nb_of_channels nb_of_frames].
            *** If delay(mic1, mic2) > 0, mic1 measure sound before (tau>0 = positive time delay (after 0))
        TDOAs_scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nTheta, nPhi]
        fs (int):
            Sample frequency

    Returns:
        Phi (np.ndarray):
            Spatial covariance matrix for each frame and bin [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    """
    nb_of_channels = TDOAs_scan.shape[0]
    nb_of_bins = f.shape[0]
    nb_of_frames = taus.shape[2]
    
    # Calculate Phi
    omega = 2*np.pi*f
    omega = np.tile(omega, (nb_of_frames, 1)).T  # [nb_of_bins, nb_of_frames]
    Phi = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels,nb_of_channels), dtype=np.csingle)
    for chan_id1 in range(nb_of_channels):
        for chan_id2 in range(nb_of_channels):
            Phi[:,:, chan_id1, chan_id2] = np.exp(1j*omega*taus[chan_id1,chan_id2,:]/fs).T
    

    return Phi

#####################################################
################ SCM_EFF CALCULATION ################
#####################################################
def compute_scm_eff_averag_with_LU(SCM_XX, SCM_VV, symmetrize=True):
    '''
    Calculate the averag effective SMC from the signal and noise SCMs.
    
    Args:
        SCM_XX (np.ndarray): 
            The TARGET spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        SCM_YY (np.ndarray):
            The NOISE spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        symmetrize (bool):
            Whether to symmetrize the effective SCM.
    
    Returns:
        SCM_eff (np.ndarray):
            The average effective SCM [nb_of_bins, nb_of_channels, nb_of_channels].
    '''
    nb_of_bins, nb_of_channels, _ = SCM_XX.shape
    
    try:
        # Diagonal loading for inversion
        SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV, shrinkage_intensity=0.001) # [nb_of_channels, nb_of_channels, nb_of_bins]

        # Compute the inverse of SCM_VV_loaded for all frequency bins
        SCM_VV_inv = np.linalg.inv(SCM_VV_loaded)  # [nb_of_bins, nb_of_channels, nb_of_channels]
        
    except np.linalg.LinAlgError as e:
        print("Error in inversion")
        SCM_VV_inv = np.tile(np.eye(nb_of_channels, dtype=np.csingle), (nb_of_bins, 1, 1))  # [nb_of_bins, nb_of_channels, nb_of_channels]
    
    SCM_eff = SCM_VV_inv @ SCM_XX  # [nb_of_bins, nb_of_channels, nb_of_channels]

    # Calculate symmetrized effective SCM
    if symmetrize:
        SCM_eff = symmetrize_matrix(SCM_eff)  # [nb_of_bins, nb_of_channels, nb_of_channels]
    
    return SCM_eff


def compute_scm_eff_with_LU(SCM_XX, SCM_VV, shrinkage_intensity=0.01, scm_inv_norm='none', symmetrize=True, eps=1e-6):
    '''
    Calculate the effective SCM from the signal and noise SCMs.
    
    Args:
        SCM_XX (np.ndarray): 
            The TARGET spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        SCM_YY (np.ndarray):
            The NOISE spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        symmetrize (bool):
            Whether to symmetrize the effective SCM.
    
    Returns:
        SCM_eff (np.ndarray):
            The effective SCM at each frame [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    '''
    nb_of_frames, nb_of_bins, nb_of_channels, _ = SCM_XX.shape
    
    # Initialize the effective SCM
    SCM_eff = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)
    
    for frame_id in range(nb_of_frames):
        
        if frame_id < nb_of_channels-1:
            # For first frames, too soon to invert -> output an Identity matrix
            SCM_VV_inv = np.tile(np.eye(nb_of_channels, dtype=np.csingle), (nb_of_bins, 1, 1))  # [nb_of_bins, nb_of_channels, nb_of_channels]
        
        else:
            # Inverse LU decomposition
            try:
                # Diagonal loading for inversion
                SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV[frame_id,...], shrinkage_intensity) # [nb_of_channels, nb_of_channels, nb_of_bins]

                # Compute the inverse of SCM_VV_loaded for all frequency bins
                SCM_VV_inv = np.linalg.inv(SCM_VV_loaded)  # [nb_of_bins, nb_of_channels, nb_of_channels]

            except np.linalg.LinAlgError as e:
                print(f"Error in frame {frame_id}: {e}")
                SCM_VV_inv = np.tile(np.eye(nb_of_channels, dtype=np.csingle), (nb_of_bins, 1, 1))  # [nb_of_bins, nb_of_channels, nb_of_channels]
            
        
        if scm_inv_norm == 'trace':
            # Normalize to prevent scaling issues
            SCM_VV_inv = SCM_VV_inv / (np.trace(SCM_VV_inv, axis1=1, axis2=2)[:, np.newaxis, np.newaxis] + eps)
            
            SCM_XX_normalized = SCM_XX[frame_id, ...] / (np.trace(SCM_XX[frame_id, ...], axis1=1, axis2=2)[:, np.newaxis, np.newaxis] + eps)  # [nb_of_bins, nb_of_channels, nb_of_channels]
        
            SCM_eff[frame_id,...] = SCM_VV_inv @ SCM_XX_normalized  # [nb_of_bins, nb_of_channels, nb_of_channels]
            
        elif scm_inv_norm == 'amplitude':
            # Normalize to prevent scaling issues
            SCM_VV_inv = SCM_VV_inv / (np.abs(SCM_VV_inv) + eps)
            
            SCM_XX_normalized = SCM_XX[frame_id, ...] / (np.abs(SCM_XX[frame_id, ...]) + eps)  # [nb_of_bins, nb_of_channels, nb_of_channels]
            
            SCM_eff[frame_id,...] = SCM_VV_inv @ SCM_XX_normalized  # [nb_of_bins, nb_of_channels, nb_of_channels]
            
        elif scm_inv_norm == 'none':
            # No normalization, use the raw inverse
            SCM_eff[frame_id,...] = SCM_VV_inv @ SCM_XX[frame_id,...]  # [nb_of_bins, nb_of_channels, nb_of_channels]

        
        # Calculate symmetrized effective SCM
        if symmetrize:
            SCM_eff[frame_id, ...] = symmetrize_matrix(SCM_eff[frame_id, ...])  # [nb_of_bins, nb_of_channels, nb_of_channels]
            
    return SCM_eff  # [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]


def compute_scm_eff_with_SM(SCM_XX, SCM_VV, V, alpha, shrinkage_intensity=0.01, scm_inv_norm='trace', symmetrize=True, nb_of_frames_to_recompute = 50, eps=1e-6):
    '''
    Sherman-Morrison formula to calculate the effective phi from the cross spectrum XX and the noise cross spectrum VV.
    
    Args:
        XX (np.ndarray): 
            The cross spectrum of target signal [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        V (np.ndarray):
            The spectrum of noise signal [nb_of_channels, nb_of_frames, nb_of_bins].
        symmetrize (bool):
            Whether to symmetrize the effective SCM.
        nb_of_frames_to_recompute (int): 
            Interval at which to recompute inverse directly
        eps (float):
            A small constant to avoid division by zero in the Sherman-Morrison update.
            
    Returns:
        SCM_eff (np.ndarray):
            The effective SCM at each frame [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    '''
    nb_of_frames, nb_of_bins, nb_of_channels, _ = SCM_XX.shape
    V = np.moveaxis(V, 0, -1)  # [nb_of_frames, nb_of_bins, nb_of_channels]
    beta = alpha / (1 - alpha)
    
    if nb_of_frames_to_recompute is None:
        nb_of_frames_to_recompute = nb_of_frames

    
    SCM_eff = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)
    SCM_VV_inv_prev = None

    
    for frame_id in range(nb_of_frames):
        recompute_by_schedule = frame_id == nb_of_channels - 1 or (frame_id % nb_of_frames_to_recompute == 0)
        
        ##### Sherman-Morrison path #####
        if frame_id < nb_of_channels - 1:
            # Too early: use only SCM_XX (SCM_VV = identity matrix)
            SCM_eff[frame_id, ...] = SCM_XX[frame_id, ...]
        
        else:
            # Compute the inverse of SCM_VV for the current frame
            if recompute_by_schedule:  
                # Inverse with LU decomposition
                try:
                    # Diagonal loading for inversion
                    SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV[frame_id,...], shrinkage_intensity) # [nb_of_channels, nb_of_channels, nb_of_bins]

                    SCM_VV_inv_new = np.linalg.inv(SCM_VV_loaded)

                except np.linalg.LinAlgError as e:
                    print(f"Error in frame {frame_id}: {e}")
                    SCM_VV_inv_new = np.tile(np.eye(nb_of_channels, dtype=np.csingle), (nb_of_bins, 1, 1))
        
            else:
                # Sherman-Morrison update (for rank-1 update)
                V_bin = V[frame_id, ...]  # [nb_of_bins, nb_of_channels]
                v = V_bin[..., np.newaxis]  # [nb_of_bins, nb_of_channels, 1]
                vH = np.conj(V_bin[:, np.newaxis, :])  # [nb_of_bins, 1, nb_of_channels]
                Phi_v = np.matmul(SCM_VV_inv_prev, v)  # [nb_of_bins, nb_of_channels, 1]
                numerator = beta * np.matmul(Phi_v, Phi_v.conj().transpose(0,2,1))  # [nb_of_bins, nb_of_channels, nb_of_channels]
                denominator = 1 + beta * np.matmul(vH, Phi_v).squeeze(-1).squeeze(-1).real  # [nb_of_bins,]

                if np.any(denominator < 1.0):
                    print(f"[Frame {frame_id}] Warning: denominator too small in Sherman-Morrison update")
                    denominator = np.maximum(denominator, 1)
                    
                SCM_VV_inv_new = 1/(1-alpha) * ( SCM_VV_inv_prev - numerator / (denominator[:, np.newaxis, np.newaxis]) )
                # SCM_VV_inv_new = SCM_VV_inv_prev

            # Normalize the inverse if needed
            if scm_inv_norm == 'trace':
                # Normalize to prevent scaling issues
                SCM_VV_inv_new = SCM_VV_inv_new / (np.trace(SCM_VV_inv_new, axis1=1, axis2=2)[:, np.newaxis, np.newaxis] + eps)
            
                SCM_XX_normalized = SCM_XX[frame_id, ...] / (np.trace(SCM_XX[frame_id, ...], axis1=1, axis2=2)[:, np.newaxis, np.newaxis] + eps)  # [nb_of_bins, nb_of_channels, nb_of_channels]
        
                # Calculate the effective SCM at the current frame
                SCM_eff[frame_id, ...] = SCM_VV_inv_new @ SCM_XX_normalized
                
            elif scm_inv_norm == 'amplitude':
                # Normalize to prevent scaling issues
                SCM_VV_inv_new = SCM_VV_inv_new / (np.abs(SCM_VV_inv_new) + eps)
                
                SCM_XX_normalized = SCM_XX[frame_id, ...] / (np.abs(SCM_XX[frame_id, ...]) + eps)
                
                # Calculate the effective SCM at the current frame
                SCM_eff[frame_id, ...] = SCM_VV_inv_new @ SCM_XX_normalized
            
            elif scm_inv_norm == 'none':
                # Calculate the effective SCM at the current frame
                SCM_eff[frame_id, ...] = SCM_VV_inv_new @ SCM_XX[frame_id, ...]
        
        
            # Calculate symmetrized effective SCM
            if symmetrize:
                SCM_eff[frame_id, ...] = symmetrize_matrix(SCM_eff[frame_id, ...])
        
            # Keep the current inverse for the next iteration
            SCM_VV_inv_prev = SCM_VV_inv_new
    
    return SCM_eff  # [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
    
    
def compute_scm_eff_with_pairwise_pseudo(SCM_XX, SCM_VV, norm_option = 'none', full_matrix=False):
    '''
    Calculate the effective phi from the signal and noise SCMs using pairwise pseudo-inverse.
    
    Args:
        SCM_XX (np.ndarray): 
            The TARGET spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        SCM_VV (np.ndarray):
            The NOISE spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        full_matrix (bool):
            Whether to return the full effective SCM matrix or just the upper triangular part.
    
    Returns:
        SCM_eff (np.ndarray):
            The symmetrized effective SCM at each frame [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    '''
    nb_of_frames, nb_of_bins, nb_of_channels, _ = SCM_XX.shape
    
    # Apply normalization if specified
    if norm_option == 'trace':
        SCM_XX = SCM_XX / (np.trace(SCM_XX, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis] + 1e-10)
        SCM_VV = SCM_VV / (np.trace(SCM_VV, axis1=2, axis2=3)[:, :, np.newaxis, np.newaxis] + 1e-10)
    elif norm_option == 'amplitude':
        SCM_XX = SCM_XX / (np.abs(SCM_XX) + 1e-15)
        SCM_VV = SCM_VV / (np.abs(SCM_VV) + 1e-15)

    # Initialize the effective SCM
    if full_matrix:
        SCM_eff = np.tile(np.eye(nb_of_channels, dtype=np.csingle), (nb_of_frames, nb_of_bins, 1, 1))
    else:
        SCM_eff = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)
    
    
    triu_indices = np.triu_indices(nb_of_channels, k=1)
    triu_i, triu_j = np.triu_indices(nb_of_channels, k=1)
    for frame_id in range(SCM_XX.shape[0]):
        if frame_id < nb_of_channels-1:
            # For first frames, too soon to invert -> output an Identity matrix
            if full_matrix:
                SCM_eff[frame_id, ...] = SCM_XX[frame_id, ...]
            else:
                SCM_eff[frame_id, :, triu_indices[0], triu_indices[1]] = SCM_XX[frame_id, :, triu_indices[0], triu_indices[1]]
            
        else:
            # Extract all needed elements for all pairs at once
            XX_ii = SCM_XX[frame_id, :, triu_i, triu_i]  # [nb_of_bins, nb_of_pairs]
            XX_ij = SCM_XX[frame_id, :, triu_i, triu_j]  # [nb_of_bins, nb_of_pairs]
            # XX_ji = SCM_XX[frame_id, :, triu_j, triu_i]  # [nb_of_bins, nb_of_pairs]
            XX_jj = SCM_XX[frame_id, :, triu_j, triu_j]  # [nb_of_bins, nb_of_pairs]

            VV_ii = SCM_VV[frame_id, :, triu_i, triu_i]  # [nb_of_bins, nb_of_pairs]
            VV_ij = SCM_VV[frame_id, :, triu_i, triu_j]  # [nb_of_bins, nb_of_pairs]
            # VV_ji = SCM_VV[frame_id, :, triu_j, triu_i]  # [nb_of_bins, nb_of_pairs]
            VV_jj = SCM_VV[frame_id, :, triu_j, triu_j]  # [nb_of_bins, nb_of_pairs]

            # Compute the pseudo-correlation for all pairs at once
            pseudo_corr = VV_jj * XX_ij - VV_ij * XX_jj - XX_ii * VV_ij + XX_ij * VV_ii  # [nb_of_bins, nb_of_pairs]

            if full_matrix:
                # Fill both (i, j) and (j, i) with conjugate
                SCM_eff[frame_id, :, triu_i, triu_j] = pseudo_corr
                SCM_eff[frame_id, :, triu_j, triu_i] = np.conj(pseudo_corr)
            else:
                SCM_eff[frame_id, :, triu_i, triu_j] = pseudo_corr      
                    
    return SCM_eff  # [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]



def compute_whitened_scm(SCM_XX, SCM_VV, shrinkage_intensity=0.01, eps=1e-8):
    """
    Compute the whitened SCM from the target and noise SCMs:
        SCM_W = SCM_VV^{-1/2} @ SCM_XX @ (SCM_VV^{-1/2})^H

    Args:
        SCM_XX (np.ndarray):
            The target spatial covariance matrix [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        SCM_VV (np.ndarray):
            The noise spatial covariance matrix [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        shrinkage_intensity (float):
            The intensity of diagonal loading for inversion.

    Returns:
        SCM_W (np.ndarray):
            The whitened SCM at each frame [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    """
    nb_of_frames, nb_of_bins, nb_of_channels, _ = SCM_XX.shape
    
    # Initialize the whitened SCM
    SCM_W = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)
    
    for frame_id in range(nb_of_frames):
        if frame_id < nb_of_channels - 1:
            # Too soon to invert, use only signal SCM
            SCM_W[frame_id] = SCM_XX[frame_id] 
        else:
            try:
                # Diagonal loading for inversion
                SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV[frame_id,...], shrinkage_intensity)

                # Eigen-decomposition for all bins at once
                eigvals, eigvecs = np.linalg.eigh(SCM_VV_loaded)  # eigvals: [nb_of_bins, nb_of_channels], eigvecs: [nb_of_bins, nb_of_channels, nb_of_channels]
                
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, eps)  # Avoid division by zero
                
                # Inverse square root of eigenvalues
                inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                
                # Diagonal matrix for inverse square root of eigenvalues
                inv_eigs_mat = np.eye(nb_of_channels, dtype=np.csingle)[np.newaxis, :, :] * inv_sqrt_eigvals[:, :, np.newaxis]
                
                # Build whitening matrices for all bins
                W = np.matmul(
                    eigvecs,
                    np.matmul(inv_eigs_mat, np.conj(np.transpose(eigvecs, (0, 2, 1))))
                )  # [nb_of_bins, nb_of_channels, nb_of_channels]
                
                # Whitening: W @ SCM_XX @ W^H for all bins
                SCM_W[frame_id] = np.matmul(
                    W,
                    np.matmul(SCM_XX[frame_id], np.conj(np.transpose(W, (0, 2, 1))))
                )

            except np.linalg.LinAlgError as e:
                print(f"Error in frame {frame_id}: {e}")
                SCM_W[frame_id] = SCM_XX[frame_id] 

    return SCM_W


def compute_approx_whitened_scm(SCM_XX, SCM_VV, shrinkage_intensity=0.01, eps=1e-8):
    """
    Compute the whitened SCM from the target and noise SCMs:
        SCM_W = SCM_VV^{-1/2} @ SCM_XX @ (SCM_VV^{-1/2})^H

    Args:
        SCM_XX (np.ndarray):
            The target spatial covariance matrix [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        SCM_VV (np.ndarray):
            The noise spatial covariance matrix [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        shrinkage_intensity (float):
            The intensity of diagonal loading for inversion.

    Returns:
        SCM_W (np.ndarray):
            The whitened SCM at each frame [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
    """
    nb_of_frames, nb_of_bins, nb_of_channels, _ = SCM_XX.shape
    
    # Initialize the whitened SCM
    SCM_W = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)
    
    for frame_id in range(nb_of_frames):
        if frame_id < nb_of_channels - 1:
            # Too soon to invert, use only signal SCM
            SCM_W[frame_id] = SCM_XX[frame_id] 
        else:
            try:
                # Diagonal loading for inversion
                SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV[frame_id,...], shrinkage_intensity)
                
                # Cholesky decomposition for all bins at once
                L = np.linalg.cholesky(SCM_VV_loaded)  # L: [nb_of_bins, nb_of_channels, nb_of_channels]
                
                # Inverse of L
                L_inv = np.linalg.inv(L)  # L_inv: [nb_of_bins, nb_of_channels, nb_of_channels]
                
                # Whitening: L @ SCM_XX @ L^H for all bins
                SCM_W[frame_id] = np.matmul(
                    L_inv,
                    np.matmul(SCM_XX[frame_id], np.conj(np.transpose(L_inv, (0, 2, 1))))
                )

            except np.linalg.LinAlgError as e:
                print(f"Error in frame {frame_id}: {e}")
                SCM_W[frame_id] = SCM_XX[frame_id] 

    return SCM_W