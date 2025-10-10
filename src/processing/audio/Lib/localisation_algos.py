import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

################################################################################
################ MAIN ALGORITHMS FOR ONE SOURCE (CALL FUNCTIONS) ###############
################################################################################
### Traditional SRP-PHAT ###
def SRP_PHAT_trad(YYs_PHAT, TDOAs_scan, f):
    '''
    Traditional SRP-PHAT algorithm with one SRP (assume one source).
    Here we use a matrix computation as the loop is not efficient in python.
    If noise, we use the effective SCM calculated from the noise and target SCMs.
    
    Args:
        YYs_PHAT (np.ndarray):
            The cross spectrum with phase transform or effective SCM [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        TDOAs_scan (np.ndarray):
            The TDOAs scan grid [nb_of_channels, nb_of_doas].
        f (np.ndarray):
            The frequency bins [nb_of_bins]. 
    Returns:
        SRP (np.ndarray):
            The Steered Response Power (SRP) for each DOA [nb_of_frames, nb_of_doas].
    '''
    # Dimensions
    nb_of_frames, _, nb_of_channels, _  = YYs_PHAT.shape
    _, nb_of_doas = TDOAs_scan.shape
    
    # Compute offline beamformer matrix
    print('Calculating beamformer matrix W...')
    W = SRP_PHAT_offline(TDOAs_scan, nb_of_channels, f)  # [nb_of_doas, nb_of_pairs*nb_of_bins]
    
    # Get the indices for the upper triangular part, excluding the diagonal
    triu_indices = np.triu_indices(nb_of_channels, k=1)
    
    # Online SRP
    time_start = time.time()
    SRP = np.zeros((nb_of_frames, nb_of_doas), dtype=np.float32)  # [nb_of_frames, nb_of_doas]
    for frame_id in tqdm(range(nb_of_frames), desc="SRP-PHAT frames"):
        # Current frame cross spectrum with phase transform
        YY_PHAT = YYs_PHAT[frame_id,...]
        
        # SRP[frame_id,:] = SRP_PHAT_online(YY_PHAT, W)
        
        # Vectorize YY_PHAT (for each frame) and remove unnecessary values, giving X [P(N/2+1),1]
        X = YY_PHAT[:, triu_indices[0], triu_indices[1]].T.flatten()  # [nb_of_pairs*nb_of_bins,]
        
        # SRP for each DOA
        SRP[frame_id,:] = np.real(W@X) # [nb_of_doas]
    
    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_per_frame = elapsed_time / nb_of_frames
    print('SRP-PHAT done in', elapsed_time, 'seconds')
    return SRP, elapsed_time_per_frame


def SVD_PHAT(YYs_PHAT, TDOAs_scan, f, delta=0.01, K=10):
    '''
    SVD-PHAT algorithm. We assume one source.
    If noise, we use the effective SCM calculated from the noise and target SCMs.
    
    Args:
        YYs_PHAT (np.ndarray):
            The cross spectrum with phase transform or effective SCM [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        TDOAs_scan (np.ndarray):
            The TDOAs scan grid [nb_of_channels, nb_of_doas].
        f (np.ndarray):
            The frequency bins [nb_of_bins]. 
    Returns:
        SRP (np.ndarray):
            The Steered Response Power (SRP) for each DOA [nb_of_frames, nb_of_doas].
    '''
    # Dimensions
    nb_of_frames, _, nb_of_channels, _  = YYs_PHAT.shape
    _, nb_of_doas = TDOAs_scan.shape
    
    # Compute offline SVD matrices : D [nb_of_doas, K], Vh_k [K, nb_of_pairs*nb_of_bins]
    print('Calculating SVD matrices...')
    # D, Vh_k = SVD_PHAT_offline_known_K(TDOAs_scan, f, K)
    D, Vh_k = SVD_PHAT_offline(TDOAs_scan, f, delta)
    
    # Get the indices for the upper triangular part, excluding the diagonal
    triu_indices = np.triu_indices(nb_of_channels, k=1)
    
    # Online SVD-PHAT
    time_start = time.time()
    SRP = np.zeros((nb_of_frames, nb_of_doas), dtype=np.float32)
    for frame_id in tqdm(range(nb_of_frames), desc="SVD-PHAT frames"):
        # Current frame cross spectrum with phase transform
        YY_PHAT = YYs_PHAT[frame_id,...]
        
        # Vectorize YY_PHAT (for each frame) and remove unnecessary values, giving X [P(N/2+1),1]
        X = YY_PHAT[:, triu_indices[0], triu_indices[1]].T.flatten()  # [nb_of_pairs*nb_of_bins,]
        
        # Calculate z
        z = Vh_k@X

        # Calculate spectrum for each source
        SRP[frame_id,:] = np.real(D@z)
        
    
    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_per_frame = elapsed_time / nb_of_frames
    print('SVD-PHAT done in', elapsed_time, 'seconds')
    return SRP, elapsed_time_per_frame


def SRP_PHAT_GCC(YYs_PHAT, TDOAs_scan, fs, q = 1, QUAD_INTERP=False):
    '''
    SRP-PHAT algorithm using GCC-PHAT. Possibility to zero pad the cross spectrum to smooth taus.
    If noise, we use the effective SCM calculated from the noise and target SCMs.
    
    Args:
        YYs_PHAT (np.ndarray):
            The cross spectrum with phase transform or effective SCM [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        TDOAs_scan (np.ndarray):
            The TDOAs scan grid [nb_of_channels, nb_of_doas].
        fs (int): 
            The sampling frequency.
        q (float):
            The zero padding factor. Default is 1 (no zero padding).
        QUAD_INTERP (bool):
            Apply parabolic interpolation.
    Returns:
        SRP (np.ndarray):
            The Steered Response Power (SRP) for each DOA [nb_of_frames, nb_of_doas].
    '''
    nb_of_frames, nb_of_bins, nb_of_channels, _  = YYs_PHAT.shape
    _, nb_of_doas = TDOAs_scan.shape
    
    # Zero pad with q factor
    FRAME_SIZE_ZERO_PAD = int((nb_of_bins-1)*2 * q)
    FS_INTERP = int(fs * q)  # Resampling frequency after zero padding
    
    # Get all unique (i, j) pairs (upper triangle, including diagonal)
    triu_i, triu_j = np.triu_indices(nb_of_channels)
    
    # Offline calculation of tdoa between microphones (in seconds)
    tau_diff = TDOAs_scan[:, np.newaxis, :] - TDOAs_scan[np.newaxis, :, :]
    
    # Gather tau indexes for all pairs and all DOAs: shape [nb_of_pairs, nb_of_doas]
    tau_diff = tau_diff[triu_i, triu_j, :]
    
    # TDOAs in samples (float or int)
    if QUAD_INTERP: # (float values)
        tau_diff_samples_float = (tau_diff * FS_INTERP)
        
        # Always take first value to the left
        tau_floor = np.floor(tau_diff_samples_float).astype(np.int32)
        delta = tau_diff_samples_float - tau_floor  # for parabolic evaluation
        
        # Take values left (farther from the true value) and right (on the other side of the true float value)
        tau_m1 = (tau_floor - 1) % FRAME_SIZE_ZERO_PAD
        tau_0  = tau_floor % FRAME_SIZE_ZERO_PAD
        tau_p1 = (tau_floor + 1) % FRAME_SIZE_ZERO_PAD
    
    else:  # take closest int value
        tau_diff_samples_int = np.rint(tau_diff * FS_INTERP).astype(np.int32) % FRAME_SIZE_ZERO_PAD
    
    # Cross correlation (GCC-PHAT)
    time_start = time.time()
    SRP = np.zeros((nb_of_frames, nb_of_doas), dtype=np.float32)  # [nb_of_frames, nb_of_doas]
    for frame_id in tqdm(range(nb_of_frames), desc="SRP-PHAT-GCC frames"):
        # Extract all relevant cross-spectra at once: shape [nb_of_pairs, nb_of_bins]
        YY_pairs = YYs_PHAT[frame_id, :, triu_i, triu_j]

        # Compute all GCCs at once: shape [nb_of_pairs, FRAME_SIZE_ZERO_PAD]
        rho_pairs = np.fft.irfft(YY_pairs, n=FRAME_SIZE_ZERO_PAD, axis=1)
        
        # Use advanced indexing to get the relevant GCC values for each pair and DOA
        if QUAD_INTERP:
            rho_m1 = np.take_along_axis(rho_pairs, tau_m1, axis=1)
            rho_0  = np.take_along_axis(rho_pairs, tau_0, axis=1)
            rho_p1 = np.take_along_axis(rho_pairs, tau_p1, axis=1)

            # Compute parablic parameters (y=ax² + bx + c)
            a = 0.5 * (rho_m1 - 2 * rho_0 + rho_p1)
            b = 0.5 * (rho_p1 - rho_m1)
            c = rho_0

            # Compute true CC value associated with float theoretical tdoa using delta
            rho_selected = a * delta**2 + b*delta + c
        
        else:
            rho_selected = np.take_along_axis(rho_pairs, tau_diff_samples_int, axis=1)  # [nb_of_pairs, nb_of_doas]
        
        # Sum over all pairs to get SRP for each DOA
        SRP[frame_id, :] = np.sum(rho_selected, axis=0)

    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_per_frame = elapsed_time / nb_of_frames
    print('SRP-PHAT-GCC done in', elapsed_time, 'seconds')
    return SRP, elapsed_time_per_frame


def SRP_PHAT_FCC(YYs_PHAT, TDOAs_scan, f, delta=0.01, K = None):
    '''
    SRP-PHAT algorithm using FCC-PHAT. 
    If noise, we use the effective SCM calculated from the noise and target SCMs.
    
    Args:
        YYs_PHAT (np.ndarray):
            The cross spectrum with phase transform or effective SCM [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        TDOAs_scan (np.ndarray):
            The TDOAs scan grid [nb_of_channels, nb_of_doas].
        f (np.ndarray):
            The frequency bins [nb_of_bins].
    Returns:
        SRP (np.ndarray):
            The Steered Response Power (SRP) for each DOA [nb_of_frames, nb_of_doas].
    '''
    nb_of_frames, nb_of_bins, nb_of_channels, _  = YYs_PHAT.shape
    _, nb_of_doas = TDOAs_scan.shape
    N = (nb_of_bins - 1) * 2
    nb_of_pairs = nb_of_channels * (nb_of_channels - 1) // 2  # Number of unique pairs (i, j)

    # Compute offline SVD matrices for each pair : Bs [nb_of_pairs, nb_of_doas, N/4+1], U_ks [nb_of_pairs, nb_of_doas, K], deltas [nb_of_pairs]
    print('Calculating SVD matrices for each pair...')
    if K is None:
        Bs, U_ks, _, K = FCC_PHAT_offline(TDOAs_scan, f, delta)
    else:
        Bs, U_ks, _ = FCC_PHAT_offline_known_K(TDOAs_scan, f, K)
        
    # Get all unique (i, j) pairs (upper triangle, including diagonal)
    triu_i, triu_j = np.triu_indices(nb_of_channels, k=1)
    
    # Get the indices for first and second half of the frequency bins
    first_half_indices = np.arange(0, N//4)
    second_half_indices = np.arange(N//2, N//4, -1)
    
    # Get even and odd vectors for z computation
    even_k = np.arange(K) % 2 == 0
    odd_k = ~even_k

    # FCC-based SRP-PHAT (FCC-PHAT)
    time_start = time.time()
    SRP = np.zeros((nb_of_frames, nb_of_doas), dtype=np.float32)  # [nb_of_frames, nb_of_doas]
    for frame_id in tqdm(range(nb_of_frames), desc="SRP-PHAT-FCC frames"):
        # Extract all relevant cross-spectra at once: shape [nb_of_pairs, nb_of_bins]
        YY_pairs = YYs_PHAT[frame_id, :, triu_i, triu_j]

        # Compute SCM_eff_s and SCM_Eff_d
        SCM_eff_s = YY_pairs[:, first_half_indices] + YY_pairs[:, second_half_indices] # [nb_of_pairs, N/4]
        SCM_eff_s = np.concatenate((SCM_eff_s, np.expand_dims(YY_pairs[:, N//4], axis=1)), axis=1)  # [nb_of_pairs, N/4+1]
        
        SCM_eff_d = YY_pairs[:, first_half_indices] - YY_pairs[:, second_half_indices]  # [nb_of_pairs, N/4]
        SCM_eff_d = np.concatenate((SCM_eff_d, np.expand_dims(YY_pairs[:, N//4], axis=1)), axis=1)  # [nb_of_pairs, N/4+1]
        
        # Compute z
        z = np.empty((nb_of_pairs, K), dtype=np.complex64)
        z[:, even_k] = np.einsum('pkj, pj -> pk', Bs[:, even_k, :], SCM_eff_s)
        z[:, odd_k]  = np.einsum('pkj, pj -> pk', Bs[:, odd_k, :], SCM_eff_d)
                
        # Compute the SRP for each DOA
        SRPs = np.real(np.einsum('pdk,pk->pd', U_ks, z))  # [nb_of_pairs, nb_of_doas]
        SRP[frame_id, :] = np.sum(SRPs, axis=0)  # Sum over pairs to get SRP for each DOA [nb_of_doas]

    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_per_frame = elapsed_time / nb_of_frames
    print('SRP-PHAT-FCC done in', elapsed_time, 'seconds')
    return SRP, elapsed_time_per_frame


def SVD_MUSIC_SCM_eff(SCM_eff, TDOAs_scan, f, eps=1e-12):
    '''
    SVD-MUSIC algorithm. We assume one source.
    
    Args:
        SCM_XX (np.ndarray):
            The TARGET spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        SCM_VV (np.ndarray):
            The NOISE spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        TDOAs_scan (np.ndarray):
            The TDOAs scan grid [nb_of_channels, nb_of_doas].
        f (np.ndarray):
            The frequency bins [nb_of_bins].
    Returns:
        SRP (np.ndarray):
            The Steered Response Power (SRP) for each DOA [nb_of_frames, nb_of_doas].
    '''
    nb_of_frames, _,_,_ = SCM_eff.shape
    _, nb_of_doas = TDOAs_scan.shape
    nb_of_sources = 1 # We always assume one source
    
    # Precalculate beamformer coefficients
    b = MUSIC_offline(TDOAs_scan, f)

    # Calculate Music spectrum
    time_start = time.time()
    SRP = np.zeros((nb_of_frames, nb_of_doas))
    for frame_id in tqdm(range(nb_of_frames), desc="SVD-MUSIC frames"):
        Mat = SCM_eff[frame_id,...]
    
        # Perform SVD on Mat. Ql: [nb_of_bins, nb_of_channels, nb_of_channels], eig_vals: [nb_of_bins, nb_of_channels]
        try:
            Ql, _, _ = np.linalg.svd(Mat)
            
        except Exception as e:
            print(f'SVD failed for frame {frame_id}: {e}')
            continue
        
        # Extract the noise subspace Qv for all frequency bins
        Qv = Ql[:, :, nb_of_sources:]  # [nb_of_bins, nb_of_channels, nb_of_channels - nb_of_sources]
        
        # Compute the projection matrix Qv * Qv^H for all frequency bins
        Proj_Qv = np.einsum('ijk,ilk->ijl', Qv, np.conj(Qv))  # [nb_of_bins, nb_of_channels, nb_of_channels]
        
        # Compute the numerator of the MUSIC spectrum for all frequency bins
        numerator = np.einsum('bmd,bmn,bnd->bd', np.conj(b), Proj_Qv, b)  # [nb_of_bins, nb_of_doas]
        
        # Music spectrum
        P_music_freq = 1 / (np.abs(numerator) + eps) # [nb_of_bins, nb_of_doas]
        SRP[frame_id,:] = np.sum(P_music_freq, axis=0) # [nb_of_doas]

    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_per_frame = elapsed_time / nb_of_frames
    print('SVD-MUSIC done in', elapsed_time, 'seconds')
    return SRP, elapsed_time_per_frame


def SVD_MUSIC(SCM_XX, SCM_VV, TDOAs_scan, f):
    '''
    SVD-MUSIC algorithm. We assume one source.
    
    Args:
        SCM_XX (np.ndarray):
            The TARGET spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        SCM_VV (np.ndarray):
            The NOISE spatial covariance matrix calculated each frame with nb_of_frames_for_average last frames [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels].
        TDOAs_scan (np.ndarray):
            The TDOAs scan grid [nb_of_channels, nb_of_doas].
        f (np.ndarray):
            The frequency bins [nb_of_bins].
    Returns:
        SRP (np.ndarray):
            The Steered Response Power (SRP) for each DOA [nb_of_frames, nb_of_doas].
    '''
    nb_of_frames, _,_,_ = SCM_XX.shape
    _, nb_of_doas = TDOAs_scan.shape
    nb_of_sources = 1 # We always assume one source
    
    # Precalculate beamformer coefficients
    b = MUSIC_offline(TDOAs_scan, f)

    # Calculate Music spectrum
    time_start = time.time()
    SRP = np.zeros((nb_of_frames, nb_of_doas))
    for frame_id in tqdm(range(nb_of_frames), desc="SVD-MUSIC frames"):
        SCM_XX_frame = SCM_XX[frame_id,...]
        SCM_VV_frame = SCM_VV[frame_id,...]
        
        # Diagonal loading of noise SCM
        SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV_frame, shrinkage_intensity=0.01)
        
        # Inverse of noise SCM
        try:
            SCM_VV_inv = np.linalg.inv(SCM_VV_loaded)
        except np.linalg.LinAlgError:
            SCM_VV_inv = np.linalg.pinv(SCM_VV_loaded)
            
        # Compute SCM_VV_inv @ SCM_XX
        Mat = np.einsum('ijk,ikl->ijl', SCM_VV_inv, SCM_XX_frame) 
    
        # Perform SVD on Mat. Ql: [nb_of_bins, nb_of_channels, nb_of_channels], eig_vals: [nb_of_bins, nb_of_channels]
        Ql, eig_vals, _ = np.linalg.svd(Mat)
        
        # Extract the noise subspace Qv for all frequency bins
        Qv = Ql[:, :, nb_of_sources:]  # [nb_of_bins, nb_of_channels, nb_of_channels - nb_of_sources]
        
        # Compute the projection matrix Qv * Qv^H for all frequency bins
        Proj_Qv = np.einsum('ijk,ilk->ijl', Qv, np.conj(Qv))  # [nb_of_bins, nb_of_channels, nb_of_channels]
        
        # Compute the numerator of the MUSIC spectrum for all frequency bins
        numerator = np.einsum('bmd,bmn,bnd->bd', np.conj(b), Proj_Qv, b)  # [nb_of_bins, nb_of_doas]
        
        # Music spectrum
        P_music_freq = 1 / np.abs(numerator) # [nb_of_bins, nb_of_doas]
        SRP[frame_id,:] = np.sum(P_music_freq, axis=0) # [nb_of_doas]

    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_per_frame = elapsed_time / nb_of_frames
    print('SVD-MUSIC done in', elapsed_time, 'seconds')
    return SRP, elapsed_time_per_frame


################################################################################
############################ LOCALISATION FUNCTIONS ############################
################################################################################

######################### CALCULATE CROSS CORRELATIONS #########################
def GCCF(YYs):
    """
    Perform Generalized Cross Correlation Function with v(f) = 1
    which is the same as the CCF in the frequency domain.
    
    Follows the GCC equations :
    
    r_y(p) = F⁻1[Psi_y(f)]
    with Psi_y(f) = v(f)phi(f)
    and with phi(f) = E[Y1^*(f)Y2(f)] = YYs

    Args:
        YYs (np.ndarray):
            The cross spectrum [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
    Returns:
        yys (np.ndarray):
            The cross correlation [nb_of_channels, nb_of_channels, nb_of_frames, frame_size]
        taus (np.ndarray):
            The maximum value of the cross correlation for each frame [nb_of_channels, nb_of_channels, nb_of_frames].
            *** If delay(mic1, mic2) > 0, mic1 measure sound before (tau>0 = positive time delay (after 0))
    """
    nb_of_frames, nb_of_bins, nb_of_channels, _ = YYs.shape
    frame_size = (nb_of_bins-1)*2

    yys = np.zeros((nb_of_channels, nb_of_channels, nb_of_frames, frame_size), dtype=np.float32)
    taus = np.zeros((nb_of_channels, nb_of_channels, nb_of_frames))
    for ii in range(nb_of_channels):
        for jj in range(nb_of_channels):
            yys[ii, jj, :, :] = np.fft.irfft(YYs[:,:,ii,jj])
            taus[ii,jj,:] = np.argmax(yys[ii,jj,:,:], axis=1)
            

    # big delays as negative delays
    taus[taus > frame_size//2] = taus[taus > frame_size//2]-frame_size

   
    
    return yys, taus


def GCC_PHAT(YYs, eps=1e-20, FRAME_SIZE_INTERP = 4096):
    """
    Perform Generalized Cross Correlation Function with Phase Transform
    using v(f) = 1/abs(Psi_y).

    Follows the GCC equations :

    r_y(p) = F⁻1[Psi_y(f)]
    with Psi_y(f) = v(f)phi(f)
    and with phi(f) = E[Y1^*(f)Y2(f)] = YYs

    Args:
        YYs (np.ndarray):
            The cross spectrum [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
        eps (float):
            Small value to prevent division by zero in Phase Transform normalisation.
        FRAME_SIZE_INTERP (int):
            Size of IFFT. If higher than signal length, the signal is zero padded ant the ifft interpolates, which gives a smoother correlation function.
    Returns:
        yys (np.ndarray):
            The cross correlation [nb_of_channels, nb_of_channels, nb_of_frames, frame_size]
        taus (np.ndarray):
            The maximum value of the cross correlation for each frame [nb_of_channels, nb_of_channels, nb_of_frames].
        (Note) If delay(mic1, mic2) > 0, mic1 measure sound before (tau>0 = positive time delay (after 0))
    """
    nb_of_frames, nb_of_bins, nb_of_channels, _ = YYs.shape
    frame_size = (nb_of_bins-1)*2

    yys = np.zeros((nb_of_channels, nb_of_channels, nb_of_frames, frame_size))
    yys_interp = np.zeros((nb_of_channels, nb_of_channels, nb_of_frames, FRAME_SIZE_INTERP))
    taus = np.zeros((nb_of_channels, nb_of_channels, nb_of_frames))
    for ii in range(nb_of_channels):
        for jj in range(nb_of_channels):
            YY = YYs[:,:,ii,jj]
            yys[ii, jj, :, :] = np.fft.irfft(YY/(np.abs(YY)+eps))
            taus[ii,jj,:] = np.argmax(yys[ii,jj,:,:], axis=1)
            yys_interp[ii,jj,:,:] = np.fft.irfft(YY/(np.abs(YY)+eps), n=FRAME_SIZE_INTERP)
            
    # big delays as negative delays
    taus[taus > frame_size//2] = taus[taus > frame_size//2]-frame_size

    return yys_interp, taus


################################ STEERED RESPONSE POWER ################################

def SRP_PHAT_offline(TDOAs_scan, nb_of_channels, f):
    '''
    Offline precalculation of beamformer matrix for Steered Response Power with Phase Transform.
    
    Args:
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        nb_of_channels (int):
            Number of channels in the array.
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,].
            
    Returns:
        W (np.ndarray):
            The beamformer matrix W [nb_of_doas, nb_of_pairs*nb_of_bins], where nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2.
    '''
    # Dimensions
    _, nb_of_doas = TDOAs_scan.shape
    nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2
    nb_of_bins = f.shape[0]
    
    # Beamformer matrix W calculation : can be precalculated offline
    TDOAs_scan = np.tile(TDOAs_scan[:,:,np.newaxis], (1, 1, nb_of_bins)) # [nb_of_channels, nb_of_doas, nb_of_bins]
    omega = 2*np.pi*f # [nb_of_bins,]
    
    W = np.zeros((nb_of_doas, nb_of_pairs*nb_of_bins), np.cdouble)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        # P_scan_1 = e^(jomega tau_i)
        Wi = np.exp(1j*omega*TDOAs_scan[chan_id1,:,:]) # [nb_of_doas, nb_of_bins]
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            # Wj = e^(jomega tau_j)
            Wj = np.exp(1j*omega*TDOAs_scan[chan_id2,:,:]) # [nb_of_doas, nb_of_bins]

            # calculate beamformer matrix Wij [nb_of_doas, (N/2+1)]
            # Wij = e^(jomega (tau_i - tau_j))
            Wij = Wi*np.conj(Wj) # [nb_of_doas, nb_of_bins]

            # W = [W1-2, W1-3, ... W15-16 ] = [[nb_of_doas, nb_of_bins], [nb_of_doas, nb_of_bins],...,[nb_of_doas, nb_of_bins]]
            W[:, pair_idx * nb_of_bins:(pair_idx + 1) * nb_of_bins] = Wij
            pair_idx += 1
            
    return W

def SRP_PHAT_online(YY_PHAT, W):
    """
    Online Steered Response Power with Phase Transform
    Performed at each frame.
    
    Here we use vectorisation because for loops are too slow in python. However, in c, the for loops might be quicker. (?)

    Args:
        YY_PHAT (np.ndarray):
            The cross spectrum with phase transform at one frame [nb_of_bins, nb_of_channels, nb_of_channels]
        W (np.ndarray):
            The beamformer matrix W [nb_of_doas, nb_of_pairs*nb_of_bins]
    Returns:
        SRP (np.ndarray):
            The steered response power with phase transorm [nb_of_doas].
    """
    _, nb_of_channels, _ = YY_PHAT.shape
    
    # Get the indices for the upper triangular part, excluding the diagonal
    triu_indices = np.triu_indices(nb_of_channels, k=1)
    
    X = YY_PHAT[:, triu_indices[0], triu_indices[1]].T.flatten()
    SRP = np.real(W@X).astype(np.float32)  # [nb_of_doas]
    
    return SRP


def SRP_PHAT_from_YYs(YYs, TDOAs_scan, f, eps=1e-20):
    """
    Perform Steered Response Power with Phase Transform in one function.

    Vectorized.

    For each frame:
        - GCC_PHAT[mic i, mic j, omega] = YYs / abs(YYs)
        - Scan_mat[omega, mic i, mic j, doa] =  exp(j omega tau(mic i, mic j, doa)) : fft-like operation where tau (TDOAs_scan) is a delay in seconds
        - SRP[doa] = sum_{mic i} sum_{mic j} sum_{omega} [ GCC_PHAT[mic i, mic j, omega] * Scan_mat[omega, mic i, mic j, doa]]

    Args:
        YYs (np.ndarray):
            The cross spectrum [nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels]
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
    Returns:
        SRP (np.ndarray):
            The steered response power with phase transorm [nb_of_frames, nb_of_doas].
    """
    nb_of_frames, nb_of_bins, nb_of_channels, _ = YYs.shape
    _, nb_of_doas = TDOAs_scan.shape
    nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2
    
    # Beamformer matrix W calculation : can be precalculated offline
    TDOAs_scan = np.tile(TDOAs_scan[:,:,np.newaxis], (1, 1, nb_of_bins)) # [nb_of_channels, nb_of_doas, nb_of_bins]
    omega = 2*np.pi*f # [nb_of_bins,]
    
    W = np.zeros((nb_of_doas, nb_of_pairs*nb_of_bins), np.cdouble)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        # P_scan_1 = e^(jomega tau_i)
        Wi = np.exp(1j*omega*TDOAs_scan[chan_id1,:,:]) # [nb_of_doas, nb_of_bins]
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            # Wj = e^(jomega tau_j)
            Wj = np.exp(1j*omega*TDOAs_scan[chan_id2,:,:]) # [nb_of_doas, nb_of_bins]

            # calculate beamformer matrix Wij [nb_of_doas, (N/2+1)]
            # Wij = e^(jomega (tau_i - tau_j))
            Wij = Wi*np.conj(Wj) # [nb_of_doas, nb_of_bins]

            # W = [W1-2, W1-3, ... W15-16 ] = [[nb_of_doas, nb_of_bins], [nb_of_doas, nb_of_bins],...,[nb_of_doas, nb_of_bins]]
            W[:, pair_idx * nb_of_bins:(pair_idx + 1) * nb_of_bins] = Wij
            pair_idx += 1


    # Online calculation
    # Get the indices for the upper triangular part, excluding the diagonal
    triu_indices = np.triu_indices(nb_of_channels, k=1)
    SRP = np.zeros((nb_of_frames, nb_of_doas), np.float32)
    for frame_id in range(nb_of_frames):
        XX = YYs[frame_id,:,:,:]
        # Apply PHAT weighting: normalize XX by its magnitude
        XX_phat = XX / (np.abs(XX) + eps)  # Adding epsilon to avoid division by zero

        # 1-vectorize XXs (for each frame) and remove unnecessary values, giving X [P(N/2+1),1]
        X = XX_phat[:, triu_indices[0], triu_indices[1]].T.flatten()

        SRP[frame_id,:] = np.real(W@X)

    return SRP


def SRP_from_GCC_PHAT(yys, TDOAs_scan, fs):
    """
    Perform Steered Response Power with Phase Transform in 2 functions (part 2).

    part 1 : yys = IFFT{GCC-PHAT} = IFFT{YYs / abs(YYs)}
    part 2 : SRP[theta, phi] = sum_{mic i} sum_{mic j} yys( tau(mic i, mic j, theta, phi) )

    Args:
        yys (np.ndarray):
            The cross correlation [nb_of_channels, nb_of_channels, nb_of_frames, frame_size]
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        fs (int):
            Sample frequency. When the IFFT in GCC-PHAT has more points (for smoothing), the sample frequency is higher.
    Returns:
        SRP (np.ndarray):
            The steered response power with phase transorm [nb_of_frames, nb_of_doas].
    """
    print('function begins')
    nb_of_channels, _, nb_of_frames, frame_size = yys.shape
    _, nb_of_doas = TDOAs_scan.shape

    SRP = np.zeros((nb_of_frames, nb_of_doas), np.float32)

    # Compute tau differences and convert to sample indices
    tau_diff = TDOAs_scan[:, np.newaxis, :] - TDOAs_scan[np.newaxis, :, :]
    tau_diff_samples = np.rint(tau_diff * fs).astype(np.int32) % frame_size
    print('loop starts')

    for chan_id1 in range(nb_of_channels):
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            print('pair:', chan_id1,chan_id2)
            R_ij = yys[chan_id1, chan_id2, :, :].astype(np.float32)  # R_ij > 0 if i measured before j [nb_of_frames, frame_size]
            
            for doa_id in range(nb_of_doas):
                SRP[:,doa_id] += R_ij[:, tau_diff_samples[chan_id1, chan_id2, doa_id]]
    
    return SRP
    


def SVD_PHAT_offline(TDOAs_scan, f, delta=0.01):
    """
    Offline precalculation for SVD-PHAT.

    From article https://arxiv.org/pdf/1906.11913
    - N = frame size
    - k = freq bin = [0,1,...,N/2] --> N/2+1 bins
    - M = number of mic
    - P = number of independent pairs = (M-1)M
    - l = frame index
    - R = number of sources
    - Q = number of possible DOAs
    
    Args:
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
    Returns:
        D (np.ndarray):
            Left projection matrix [nb_of_doas, K], where K is respects the delta condition.
        Vh_k (np.ndarray):
            Right projection matrix [K, nb_of_pairs*nb_of_bins]
    """
    nb_of_channels, nb_of_doas = TDOAs_scan.shape
    nb_of_bins = f.shape[0]
    nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2
    
    # Calculate beamformer matrix W [nb_of_doas, nb_of_pairs*nb_of_bins]
    TDOAs_scan = np.tile(TDOAs_scan[:,:,np.newaxis], (1, 1, nb_of_bins)) # [nb_of_channels, nb_of_doas, nb_of_bins]
    omega = 2*np.pi*f # [nb_of_bins,]
    W = np.zeros((nb_of_doas, nb_of_pairs*nb_of_bins), np.cdouble)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        # P_scan_1 = e^(jomega tau_i)
        Wi = np.exp(1j*omega*TDOAs_scan[chan_id1,:,:]) # [nb_of_doas, nb_of_bins]
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            # Wj = e^(jomega tau_j)
            Wj = np.exp(1j*omega*TDOAs_scan[chan_id2,:,:]) # [nb_of_doas, nb_of_bins]

            # calculate beamformer matrix Wij [nb_of_doas, (N/2+1)]
            # Wij = e^(jomega (tau_i - tau_j))
            Wij = Wi*np.conj(Wj) # [nb_of_doas, nb_of_bins]

            W[:, pair_idx * nb_of_bins:(pair_idx + 1) * nb_of_bins] = Wij
            pair_idx += 1

    # Compute the SVD of W
    U, s, Vh = np.linalg.svd(W, full_matrices=False)
    
    # Compute the cumulative energy of singular values
    cumulative_energy = np.cumsum(s**2)
    total_energy = cumulative_energy[-1]
    
    # Find the smallest K that satisfies the condition
    K = np.searchsorted(cumulative_energy, (1 - delta) * total_energy) + 1
    
    # Construct the rank-K approximation of W
    U_k = U[:, :K]
    S_k = np.diag(s[:K])
    Vh_k = Vh[:K, :]
    D = U_k@S_k
    # W_approx = np.dot(U_k, np.dot(S_k, Vh_k))

    return D, Vh_k


def SVD_PHAT_offline_known_K(TDOAs_scan, f, K=10):
    """
    Offline precalculation for SVD-PHAT with known K.

    From article https://arxiv.org/pdf/1906.11913
    
    Args:
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
        K (int):
            K-rank approximation of W matrix
    Returns:
        D (np.ndarray):
            Left projection matrix [nb_of_doas, K], where K is respects the delta condition.
        Vh_k (np.ndarray):
            Right projection matrix [K, nb_of_pairs*nb_of_bins]
    """
    nb_of_channels, nb_of_doas = TDOAs_scan.shape
    nb_of_bins = f.shape[0]
    nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2
    
    # Calculate beamformer matrix W [nb_of_doas, nb_of_pairs*nb_of_bins]
    TDOAs_scan = np.tile(TDOAs_scan[:,:,np.newaxis], (1, 1, nb_of_bins)) # [nb_of_channels, nb_of_doas, nb_of_bins]
    omega = 2*np.pi*f # [nb_of_bins,]
    W = np.zeros((nb_of_doas, nb_of_pairs*nb_of_bins), np.cdouble)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        # P_scan_1 = e^(jomega tau_i)
        Wi = np.exp(1j*omega*TDOAs_scan[chan_id1,:,:]) # [nb_of_doas, nb_of_bins]
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            # Wj = e^(jomega tau_j)
            Wj = np.exp(1j*omega*TDOAs_scan[chan_id2,:,:]) # [nb_of_doas, nb_of_bins]

            # calculate beamformer matrix Wij [nb_of_doas, (N/2+1)]
            # Wij = e^(jomega (tau_i - tau_j))
            Wij = Wi*np.conj(Wj) # [nb_of_doas, nb_of_bins]

            W[:, pair_idx * nb_of_bins:(pair_idx + 1) * nb_of_bins] = Wij
            pair_idx += 1

    # Compute the SVD of W
    U, s, Vh = np.linalg.svd(W, full_matrices=False)
    
    # Construct the rank-K approximation of W
    U_k = U[:, :K]
    S_k = np.diag(s[:K])
    Vh_k = Vh[:K, :]
    D = U_k@S_k
    # W_approx = np.dot(U_k, np.dot(S_k, Vh_k))

    return D, Vh_k

def SVD_PHAT_online(XX, D, Vh_k, nb_of_sources=2):
    '''
    Online calculation for SVD-PHAT, using observations (cross spectrum) and precalculated matrices.

    From article https://arxiv.org/pdf/1906.11913
    - N = frame size
    - k = freq bin = [0,1,...,N/2] --> N/2+1 bins
    - M = number of mic
    - P = number of independent pairs = (M-1)M
    - l = frame index
    - R = number of sources
    - Q = number of possible DOAs

    Args:
        XX (np.ndarray):
            The cross spectrum at one frame [nb_of_bins, nb_of_channels, nb_of_channels]
        D (np.ndarray):
            Left projection matrix [nb_of_doas, K], where K is respects the delta condition.
        Vh_k (np.ndarray):
            Right projection matrix [K, nb_of_pairs*nb_of_bins]
    Returns:
        DOAs_id (np.array): 
            Detected peak id for given frame for each source [nb_of_sources].
    """

    '''
    _, nb_of_channels,_ = XX.shape
    _, K = D.shape
    # nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2

    # Get the indices for the upper triangular part, excluding the diagonal
    triu_indices = np.triu_indices(nb_of_channels, k=1)

    # Apply PHAT weighting: normalize XX by its magnitude
    XX_phat = XX / (np.abs(XX) + 1e-20)  # Adding epsilon to avoid division by zero

    # Vectorize XXs (for each frame) and remove unnecessary values, giving X [P(N/2+1),1]
    X = XX_phat[:, triu_indices[0], triu_indices[1]].T.flatten()

    # Initialize the orthogonal basis
    U = np.zeros((nb_of_sources, K), dtype=complex)

    # Calculate Z
    Z = Vh_k@X

    # P = np.zeros((nb_of_sources, nb_of_doas), np.float32)
    DOAs_id = np.zeros((nb_of_sources), np.int32)
    for source_id in range(nb_of_sources):
        # Calculate spectrum for each source (each iteration, Z is updated without the previous max source contribution)
        P = np.real(D@Z)

        # Max id corresponding to DOA
        DOAs_id[source_id] = np.argmax(P)

        # vr
        vr = D[DOAs_id[source_id],:]  # [1, K]

        # Orthogonalize vr against all previously found vectors using Gram-Schmidt
        ur = vr.copy()
        for n in range(source_id):
            projection = np.dot(U[n, :], vr) * U[n, :]
            ur -= projection

        # Normalize ur to obtain ûr
        norm_ur = np.linalg.norm(ur)
        if norm_ur > 0:
            U[source_id, :] = ur / norm_ur
        else:
            raise ValueError(f"Zero norm encountered at iteration {source_id}, check input data.")

        # Project Z onto ûr and remove this component from Z
        projection_Z = np.dot(U[source_id, :], Z) * U[source_id, :].conj().T
        Z -= projection_Z

    return DOAs_id


def FCC_PHAT_offline(TDOAs_scan, f, delta=0.01):
    '''
    Offline precalculation for FCC-PHAT.
    From article https://arxiv.org/pdf/1906.11913
    
    Args:
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
        delta (float):
            The maximum tolerated ratio of energy loss for each pair.
    Returns:
        Bs (np.ndarray):
            The beamformer matrix Bs [nb_of_pairs, nb_of_doas, N/4+1], where N = nb_of_samples.
        U_ks (np.ndarray):
            The left projection matrix U_ks [nb_of_pairs, nb_of_doas, K].
        deltas (np.ndarray):
            The ratio of energy for each pair [nb_of_pairs].
        K_global (int):
            The global K used for all pairs.
    '''
    nb_of_channels, nb_of_doas = TDOAs_scan.shape
    nb_of_bins = f.shape[0]
    nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2
    N = (nb_of_bins - 1) * 2
    
    # Calculate beamformer matrices W separatly for each pair [nb_of_pairs, nb_of_doas, nb_of_bins]
    TDOAs_scan = np.tile(TDOAs_scan[:,:,np.newaxis], (1, 1, nb_of_bins)) # [nb_of_channels, nb_of_doas, nb_of_bins]
    omega = 2*np.pi*f # [nb_of_bins,]
    
    # ---------- PASS 1: determine per-pair K_i ----------
    min_dim = min(nb_of_doas, nb_of_bins)
    U_all = np.zeros((nb_of_pairs, nb_of_doas, min_dim), dtype=np.complex128)
    s_all = np.zeros((nb_of_pairs, min_dim), dtype=np.float64)
    Vh_all = np.zeros((nb_of_pairs, min_dim, nb_of_bins), dtype=np.complex128)
    Ks = np.zeros(nb_of_pairs, dtype=np.int32)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        Wi = np.exp(1j*omega*TDOAs_scan[chan_id1,:,:]) # [nb_of_doas, nb_of_bins]
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            Wj = np.exp(1j*omega*TDOAs_scan[chan_id2,:,:]) # [nb_of_doas, nb_of_bins]

            # calculate beamformer matrix Wij [nb_of_doas, (N/2+1)]
            # Wij = e^(jomega (tau_i - tau_j))
            Wij = Wi*np.conj(Wj) # [nb_of_doas, nb_of_bins]

            # SVD on each pair
            # U, s, Vh = np.linalg.svd(Wij, full_matrices=False)
            U_all[pair_idx], s_all[pair_idx], Vh_all[pair_idx] = np.linalg.svd(Wij, full_matrices=False)
            
            # compute cumulative energy ratio
            energy = np.cumsum(s_all[pair_idx]**2) / np.sum(s_all[pair_idx]**2)
            Ks[pair_idx] = np.searchsorted(energy, (1-delta)) + 1  # +1 since indices start at 0
            
            pair_idx += 1
            
    # ---------- determine global K ----------
    K_global = max(Ks)

    # ---------- PASS 2: build truncated approximations ----------
    Bs = np.zeros((nb_of_pairs, K_global, N//4+1), np.cdouble) # [nb_of_pairs, nb_of_doas, N/4+1]
    U_ks = np.zeros((nb_of_pairs, nb_of_doas, K_global), np.cdouble) # [nb_of_pairs, nb_of_doas, K]
    deltas = np.zeros((nb_of_pairs), np.float32)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        for chan_id2 in range(chan_id1+1, nb_of_channels):
    
            # Construct the rank-K approximation of W
            U_ks[pair_idx,...] = U_all[pair_idx, :, :K_global]
            S_k = np.diag(s_all[pair_idx, :K_global])
            Vh_k = Vh_all[pair_idx, :K_global, :]
            
            # Compute B and only keep the first half of Bs freq bins
            B = S_k @ Vh_k # [K, nb_of_bins]
            Bs[pair_idx,...] = B[:,:N//4+1] # [K, nb_of_bins//2+1]
            
            # Verification of deltas (reconstruction error)
            cumulative_energy_SVD = np.cumsum(S_k**2) / np.sum(s_all[pair_idx]**2)
            deltas[pair_idx] = 1 - cumulative_energy_SVD[-1]
            
            pair_idx += 1
        
    return Bs, U_ks, deltas, K_global


def FCC_PHAT_offline_known_K(TDOAs_scan, f, K):
    '''
    Offline precalculation for FCC-PHAT.
    From article https://arxiv.org/pdf/1906.11913
    
    Args:
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
        K (int):
            Size of the rank-K approximation to be used in FCC-PHAT.
    Returns:
        Bs (np.ndarray):
            The beamformer matrix Bs [nb_of_pairs, nb_of_doas, N/4+1], where N = nb_of_samples.
        U_ks (np.ndarray):
            The left projection matrix U_ks [nb_of_pairs, nb_of_doas, K].
        deltas (np.ndarray):
            The ratio of energy for each pair [nb_of_pairs].
    '''
    nb_of_channels, nb_of_doas = TDOAs_scan.shape
    nb_of_bins = f.shape[0]
    nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2
    N = (nb_of_bins - 1) * 2
    
    # Calculate beamformer matrices W separatly for each pair [nb_of_pairs, nb_of_doas, nb_of_bins]
    TDOAs_scan = np.tile(TDOAs_scan[:,:,np.newaxis], (1, 1, nb_of_bins)) # [nb_of_channels, nb_of_doas, nb_of_bins]
    omega = 2*np.pi*f # [nb_of_bins,]
    # W = np.zeros((nb_of_pairs, nb_of_doas, nb_of_bins), np.cdouble)
    Bs = np.zeros((nb_of_pairs, K, N//4+1), np.cdouble) # [nb_of_pairs, nb_of_doas, N/4+1]
    U_ks = np.zeros((nb_of_pairs, nb_of_doas, K), np.cdouble) # [nb_of_pairs, nb_of_doas, K]
    deltas = np.zeros((nb_of_pairs), np.float32)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        # P_scan_1 = e^(jomega tau_i)
        Wi = np.exp(1j*omega*TDOAs_scan[chan_id1,:,:]) # [nb_of_doas, nb_of_bins]
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            # Wj = e^(jomega tau_j)
            Wj = np.exp(1j*omega*TDOAs_scan[chan_id2,:,:]) # [nb_of_doas, nb_of_bins]

            # W[pair_idx, :, :] = Wij
            # calculate beamformer matrix Wij [nb_of_doas, (N/2+1)]
            # Wij = e^(jomega (tau_i - tau_j))
            Wij = Wi*np.conj(Wj) # [nb_of_doas, nb_of_bins]

            # SVD on each pair
            U, s, Vh = np.linalg.svd(Wij, full_matrices=False)
    
            # Compute the cumulative energy of singular values
            cumulative_energy = np.cumsum(s**2)
            total_energy = cumulative_energy[-1]
    
            # Construct the rank-K approximation of W
            U_ks[pair_idx,...] = U[:, :K]
            S_k = np.diag(s[:K])
            Vh_k = Vh[:K, :]
            B = S_k @ Vh_k # [K, nb_of_bins]
            
            # Compute the cumulative energy of singular values
            cumulative_energy_SVD = np.cumsum(S_k**2)
            total_energy_SVD = cumulative_energy_SVD[-1]
            
            # Compute ratio of energy
            deltas[pair_idx] = total_energy_SVD / total_energy
            
            # Only keep the first half of Bs freq bins
            Bs[pair_idx,...] = B[:,:N//4+1] # [K, nb_of_bins//2+1]
            
            
            pair_idx += 1
            
    return Bs, U_ks, deltas

#################################################################################
################################  LOCALISATION   ################################
#################################################################################

def DOAs_id_from_SRP(SRP):
    '''
    Calculate maximum DOAs from SRP.
    
    Multiple sources option not possible here because it is a doa vector.
    Must use same function in localisation_theta_phi.py to use gradient type method to remove the peak.

    Args:
        SRP (np.ndarray): 
            Steered Response Power spectrum [nb_of_frames, nb_of_doas]
    
    Returns:
        DOAs_id (np.array): 
            Detected peak id for each frame [nb_of_frames].

    '''
    if len(SRP.shape) == 1:
        SRP = np.expand_dims(SRP, axis=0)

    nb_of_frames=SRP.shape[0]

    DOAs_id = np.zeros((nb_of_frames), dtype=np.int32)
    for frame_id in range(nb_of_frames):
        DOAs_id[frame_id] = np.argmax(SRP[frame_id,:])

    return DOAs_id


def DOAs_coordinate_from_DOAs_id(DOAs_id, spherical_grid):
    '''
    Calculate maximum DOAs from SRP.
    
    Multiple sources option not possible here because it is a doa vector.
    Must use same function in localisation_theta_phi.py to use gradient type method to remove the peak.

    Args:
        DOAs_id (np.array): 
            Detected peak id for each frame [nb_of_frames]
        spherical_grid (np.ndarray):
            Spherical grid of radius = 1 [nb_of_points, 3].
    
    Returns:
        DOAs_coordinates (np.array): 
            Detected peak coordinates (x,y,z) for each frame [nb_of_frames, 3].

    '''
    nb_of_frames = DOAs_id.shape[0]

    DOAs_coordinates = np.zeros((nb_of_frames, 3))
    for frame_id in range(nb_of_frames):
        DOAs_coordinates[frame_id,:] = spherical_grid[DOAs_id[frame_id],:]

    return DOAs_coordinates

def DOAs_from_DOAs_coordinates(DOAs_coordinates):
    '''
    Convert Cartesian coordinates (x, y, z) to spherical angles (theta, phi) in degrees.

    Args:
        DOAs_coordinates (np.array): 
            Detected peak coordinates (x, y, z) for each frame [nb_of_frames, 3]

    Returns:
        DOAs (np.ndarray):
            Detected DOAs (theta, phi) in degrees [nb_of_frames, 2]
    '''
    x, y, z = DOAs_coordinates[:, 0], DOAs_coordinates[:, 1], DOAs_coordinates[:, 2]
    
    # Compute azimuth angle theta (longitude)
    theta = np.arctan2(y, x)  # Ranges from -π to π
    
    # Compute elevation angle phi (latitude)
    phi = np.arccos(z)  # Ranges from 0 to π

    # Convert radians to degrees
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    # Adjust theta to range from 0 to 360 degrees
    theta_deg = np.where(theta_deg < 0, theta_deg + 360, theta_deg)

    DOAs = np.vstack((theta_deg, phi_deg)).T
    return DOAs

def filter_doa_by_phi(doas):
    """
    Filters DOAs to select the most relevant source per frame based on elevation angle φ.
    
    Now just for 2 sources, and φ < 90°

    Parameters:
    doas (np.ndarray): Array of shape [nb_of_sources, nb_of_frames, 2], where each DOA is (theta, phi) in degrees.

    Returns:
    np.ndarray: Array of shape [nb_of_frames, 2] with the selected DOAs per frame. Frames with no valid source have NaN values.
    """
    nb_of_sources, nb_of_frames, _ = doas.shape
    output = np.full((nb_of_frames, 2), np.nan)

    for frame_id in range(nb_of_frames):
        # Check if the first source has φ < 90°
        if doas[0, frame_id, 1] < 90:
            output[frame_id] = doas[0, frame_id]
        # Check if the second source exists and has φ < 90°
        elif nb_of_sources > 1 and doas[1, frame_id, 1] < 90:
            output[frame_id] = doas[1, frame_id]
        # If neither source has φ < 90°, the output remains NaN
    return output

def remove_source_peak_from_gcc(DOAs_id, TDOAs_scan, yys, fs):
    '''
    Remove peak in GCC associated with maximum source using gradient method.

    Args:
        DOAs_id (np.array): 
            List of detected peak id for each frame [nb_of_frames]
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        yys (np.ndarray):
            The cross correlation [nb_of_channels, nb_of_channels, nb_of_frames, frame_size]
        fs (int):
            Sample frequency. When the IFFT in GCC-PHAT has more points (for smoothing), the sample frequency is higher.

    Returns:
        yys_modified (np.ndarray):
            The modified cross correlation without the maximum source [nb_of_channels, nb_of_channels, nb_of_frames, frame_size].
    '''
    nb_of_channels, _, nb_of_frames, frame_size = yys.shape
    _, nb_of_doas = TDOAs_scan.shape
    yys = yys.astype(np.float32)

    # Shift cross-correlation to 0 delay = center
    yys_shifted = np.roll(yys, shift=frame_size // 2, axis=-1)
    # yys_shifted = np.zeros(yys.shape)
    # yys_shifted[:,:,:,:frame_size//2] = yys[:,:,:,frame_size//2:]
    # yys_shifted[:,:,:,frame_size//2:] = yys[:,:,:,:frame_size//2]
    

    yys_modified_shifted = yys_shifted.copy()
    for frame_id in range(nb_of_frames):
        # free field tdoa associated with DOA from SRP-PHAT
        TDOAs_max_SRP = TDOAs_scan[:,DOAs_id[frame_id]]*fs # [nb_of_channels,]

        for chan_id1 in range(nb_of_channels):
            for chan_id2 in range(nb_of_channels):
                # Cross corr for each pair of mics
                yys_ij_frame = yys_shifted[chan_id1, chan_id2, frame_id, :]  # [frame_size]

                # Theoretical taus
                TDOA_diff_SRP = int(np.round(TDOAs_max_SRP[chan_id1] - TDOAs_max_SRP[chan_id2]))+frame_size//2

                # Find max in yys_shifted
                peak_found = False
                yys_peak_id = TDOA_diff_SRP
                while not peak_found:
                    yys_peak_val = yys_ij_frame[yys_peak_id]
                    # if max, values before and after are lower
                    if yys_peak_val >= yys_ij_frame[yys_peak_id-1]:
                        if yys_peak_val >= yys_ij_frame[yys_peak_id+1]:
                            peak_found = True
                        else: # max is after currnt yys_peak_id
                            yys_peak_id += 1
                    else: # max is before currnt yys_peak_id
                        yys_peak_id -= 1

                # Null peak value
                yys_modified_shifted[chan_id1, chan_id2, frame_id, yys_peak_id] = 0

                # Null the peak left
                left_neg_gradient = True
                yys_id = yys_peak_id
                while left_neg_gradient:
                    # if value left is lower, set to zero and continue
                    ref_val = yys_ij_frame[yys_id]
                    left_val = yys_ij_frame[yys_id-1]
                    if left_val <= ref_val:
                        yys_modified_shifted[chan_id1, chan_id2, frame_id, yys_id-1] = 0
                        yys_id -= 1
                    else:
                        left_neg_gradient = False

                # Null the peak right
                right_neg_gradient = True
                yys_id = yys_peak_id
                while right_neg_gradient:
                    # if value right is lower, set to zero and continue
                    ref_val = yys_ij_frame[yys_id]
                    right_val = yys_ij_frame[yys_id+1]
                    if right_val <= ref_val:
                        yys_modified_shifted[chan_id1, chan_id2, frame_id, yys_id+1] = 0
                        yys_id += 1
                    else:
                        right_neg_gradient = False

    # Shift cross-correlation to 0 delay = 0
    yys_modified = np.roll(yys_modified_shifted, shift=-frame_size // 2, axis=-1)
    # yys_modified = np.zeros(yys_modified_shifted.shape)
    # yys_modified[:,:,:,:frame_size//2] = yys_modified_shifted[:,:,:,frame_size//2:]
    # yys_modified[:,:,:,frame_size//2:] = yys_modified_shifted[:,:,:,:frame_size//2]

    return yys_modified


#################################################################################
###################################   MUSIC    ##################################
#################################################################################

def GEVD_MUSIC(SCM_YY, SCM_VV, TDOAs_Scan, f, nb_of_sources = 0, eigenvalue_threshold = 100):
    """
    Localise the direction(s) of arrival of the sound source(s) for 3D array using GEVD-MUSIC algorithm.
    Can work both offline or online, as it uses spatial covariance matrices.
    If used online, call the function iteratively at each frame providing the SCMs at each frame.
    If used offline, with a fixed sources position, call the function one time with the SCM, which will give one position.

    Args:
        SCM_YY (np.ndarray):
            The spatial covariance matrix of the measured signal [nb_of_channels, nb_of_channels, nb_of_bins]
        SCM_VV (np.ndarray):
            The spatial covariance matrix of the noise [nb_of_channels, nb_of_channels, nb_of_bins]
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nTheta, nPhi]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
        nb_of_sources (int):
            Number of sources if it is known, 0 if it not.
        eigenvalue_threshold (float):
            If number of sources not known, threshold to separate sources subspace with noise subspace.
    Returns:
        P_music (np.ndarray):
            The total MUSIC pseudo-spectrum combining all frequency bins [nTheta, nPhi].
    """
    _, _, nb_of_bins = SCM_YY.shape
    _, nTheta, nPhi = TDOAs_Scan.shape

    # Change dimensions (éventuellement normaliser les SCM comme ça dans toutes ls fonctions, plus intuitif...)
    SCM_VV = SCM_VV.transpose(2, 0, 1)  # [nb_of_bins, nb_of_channels, nb_of_channels]
    SCM_YY = SCM_YY.transpose(2, 0, 1)  # [nb_of_bins, nb_of_channels, nb_of_channels]
    
    # Diagonal loading for inversion
    SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV, shrinkage_intensity=0.01) # [nb_of_channels, nb_of_channels, nb_of_bins]
    
    P_music = np.zeros((nTheta, nPhi), dtype=np.float32)
    for bin_id in range(nb_of_bins):
        # Eigenvalue decomposition of signal with noise whitening
        Mat = np.linalg.inv(SCM_VV_loaded[bin_id,:,:])@SCM_YY[bin_id,:,:]
        eig_val, eig_vec = np.linalg.eig(Mat)

        if nb_of_sources == 0:
            nb_of_sources = (np.abs(eig_val)>eigenvalue_threshold).sum()
            if nb_of_sources == 0:
                nb_of_sources = 1

        # Qv
        Qv = eig_vec[:,nb_of_sources:]  # [nb_of_channels, nb_of_channels-nb_of_sources]

        # DOA candidate vector
        b = np.exp(-1j*2*np.pi*f[bin_id]*TDOAs_Scan)  # [nb_of_channels, nTheta, nPhi]
        b_H = np.conj(b).transpose(1, 2, 0)  # [nTheta, nPhi, nb_of_channels]

        # Compute the projection matrix Qv * Qv^H
        Proj_Qv = Qv @ np.conj(Qv).T  # [nb_of_channels, nb_of_channels]

        # Compute the numerator of the MUSIC spectrum
        numerator = np.einsum('ijk,kl,lij->ij', b_H, Proj_Qv, b)  # [nTheta, nPhi]

        # Music spectrum
        P_music += 1 / np.abs(numerator)

    return P_music


def MUSIC_offline(TDOAs_Scan, f):
    """
    Offline computation of the beamformer coefficients to use in SVD, GSVD or GEVD MUSIC.
    
    Args:
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
    Returns:
        b (np.ndarray):
            The beamformer coefficients [nb_of_bins, nb_of_channels, nb_of_doas]
    """
    # Compute the steering vector b for all frequency bins
    omega = 2 * np.pi * f[:, np.newaxis, np.newaxis]  # [nb_of_bins, 1, 1]
    b = np.exp(-1j * omega * TDOAs_Scan[np.newaxis, :, :])  # [nb_of_bins, nb_of_channels, nb_of_doas]
    
    return b

def GSVD_MUSIC_vec(SCM_YY, SCM_VV, b, nb_of_sources = 0, singlevalue_threshold = 2):
    """
    Localise the direction(s) of arrival of the sound source(s) for 3D array using SVD-MUSIC algorithm.*** SVD and not GSVD here ***
    The frequency loop was replaced with a matrix multiplication, which makes it faster.
    Can work both offline or online, as it uses spatial covariance matrices.
    If used online, call the function iteratively at each frame providing the SCMs at each frame.
    If used offline, with a fixed sources position, call the function one time with the SCM, which will give one position.

    Args:
        SCM_YY (np.ndarray):
            The spatial covariance matrix of the measured signal [nb_of_bins, nb_of_channels, nb_of_channels]
        SCM_VV (np.ndarray):
            The spatial covariance matrix of the noise [nb_of_bins, nb_of_channels, nb_of_channels]
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
        nb_of_sources (int):
            Number of sources if it is known, 0 if it not.
        singlevalue_threshold (float):
            If number of sources not known, threshold to separate sources subspace with noise subspace.
    Returns:
        P_music (np.ndarray):
            The total MUSIC pseudo-spectrum combining all frequency bins [nb_of_doas]
        P_music_freq (np.ndarray):
            (optional, add manualy) The MUSIC pseudo-spectrum at each frequency bin [nb_of_bins. nb_of_doas].
    """
    start = time.time()

    # Diagonal loading for inversion
    SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV, shrinkage_intensity=0.1) # [nb_of_channels, nb_of_channels, nb_of_bins]

    # Compute the inverse of SCM_VV_loaded for all frequency bins
    try:
        SCM_VV_inv = np.linalg.inv(SCM_VV_loaded)   # [nb_of_bins, nb_of_channels, nb_of_channels]
    except np.linalg.LinAlgError:
        SCM_VV_inv = np.linalg.pinv(SCM_VV_loaded)

    mat_inv = time.time()

    # Compute the product of SCM_VV_inv and SCM_YY for all frequency bins
    Mat = np.einsum('ijk,ikl->ijl', SCM_VV_inv, SCM_YY)  # [nb_of_bins, nb_of_channels, nb_of_channels]
    mat_calc = time.time()
    
    # Perform SVD on Mat for all frequency bins
    Ql, eig_vals, _ = np.linalg.svd(Mat)  # Ql: [nb_of_bins, nb_of_channels, nb_of_channels], eig_vals: [nb_of_bins, nb_of_channels]
    sing_val_decomp = time.time()

    # Determine the number of sources if not provided
    if nb_of_sources == 0:
        nb_of_sources = np.sum(eig_vals > singlevalue_threshold, axis=0)
        nb_of_sources[nb_of_sources == 0] = 1  # Ensure at least one source is assumed
    
    # Extract the noise subspace Qv for all frequency bins
    Qv = Ql[:, :, nb_of_sources:]  # [nb_of_bins, nb_of_channels, nb_of_channels - nb_of_sources]

    # Compute the projection matrix Qv * Qv^H for all frequency bins
    Proj_Qv = np.einsum('ijk,ilk->ijl', Qv, np.conj(Qv))  # [nb_of_bins, nb_of_channels, nb_of_channels]
    proj_qv  = time.time()



    # Compute the numerator of the MUSIC spectrum for all frequency bins
    numerator = np.einsum('bmd,bmn,bnd->bd', np.conj(b), Proj_Qv, b)  # [nb_of_bins, nb_of_doas]
    big_mat_calc=time.time()

    # Music spectrum
    P_music_freq = 1 / np.abs(numerator) # [nb_of_bins, nb_of_doas]
    P_music = np.sum(P_music_freq, axis=0) # [nb_of_doas]
    end_time= time.time()

    # print('mat inv (common for both)',mat_inv-start)
    # print('mat calculation (common for both)',mat_calc-mat_inv)
    # print('single value decomposition (unique)',sing_val_decomp-mat_calc)
    # print('projction of qv with itself (unique)',proj_qv-sing_val_decomp)
    # print('calculation of b (common for both)',b_calc-proj_qv)
    # print('big matrix calculation (common for both)',big_mat_calc-b_calc)
    # print('end calculatoin (common for both)',end_time-big_mat_calc)
    # print('total calculatoin for music',end_time-start)

    return P_music





#################################################################################
##########################   MVDR-Based localisation    #########################
#################################################################################

def localisation_mvdr_vec(SCM_YY, SCM_VV, TDOAs_Scan, f, eps=1e-6):
    """
    Localise the direction(s) of arrival of the sound source(s) for 3D array using "MVDR"-based localisation algorithm.
    Can work both offline or online, as it uses spatial covariance matrices.
    If used online, call the function iteratively at each frame providing the SCMs at each frame.
    If used offline, with a fixed sources position, call the function one time with the SCM, which will give one position.

    Args:
        SCM_YY (np.ndarray):
            The spatial covariance matrix of the measured signal [nb_of_bins, nb_of_channels, nb_of_channels]
        SCM_VV (np.ndarray):
            The spatial covariance matrix of the noise [nb_of_bins, nb_of_channels, nb_of_channels]
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,].
    Returns:
        P_mvdr (np.ndarray):
            The total pseudo-spectrum combining all frequency bins [nb_of_doas]
        P_mvdr_freq (np.ndarray):
            (optional, add manualy) The pseudo-spectrum at each frequency bin [nb_of_doas, nb_of_bins].
    """
    start = time.time()
    # Diagonal loading for inversion
    SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV, shrinkage_intensity=0.01) # [nb_of_channels, nb_of_channels, nb_of_bins]

    # Compute the inverse of SCM_VV_loaded for all frequency bins
    SCM_VV_inv = np.linalg.inv(SCM_VV_loaded)  # [nb_of_bins, nb_of_channels, nb_of_channels]

    # Compute the product of SCM_VV_inv and SCM_YY for all frequency bins
    Mat = np.einsum('ijk,ikl->ijl', SCM_VV_inv, SCM_YY)  # [nb_of_bins, nb_of_channels, nb_of_channels]

    # Compute the steering vector b for all frequency bins
    omega = 2 * np.pi * f[:, np.newaxis, np.newaxis]  # [nb_of_bins, 1, 1]
    b = np.exp(-1j * omega * TDOAs_Scan[np.newaxis, :, :])  # [nb_of_bins, nb_of_channels, nb_of_doas]

    # Compute the MVDR-like spectrum for all frequency bins
    P_mvdr_freq = np.abs(np.einsum('bmd,bmn,bnd->bd', np.conj(b), Mat, b))  # [nb_of_bins, nb_of_doas]

    # MVDR-like spectrum
    P_mvdr = np.sum(P_mvdr_freq, axis=0) # [nb_of_doas]

    end_time = time.time()
    print('total calculatoin for MVDR',end_time-start)

    return P_mvdr


def localisation_mvdr_pairwise(SCM_YY, SCM_VV, W, f, eps=1e-6):
    """
    Localise the direction(s) of arrival of the sound source(s) for 3D array using "MVDR"-based localisation algorithm.
    Can work both offline or online, as it uses spatial covariance matrices.
    If used online, call the function iteratively at each frame providing the SCMs at each frame.
    If used offline, with a fixed sources position, call the function one time with the SCM, which will give one position.

    P = nb_of_channels * (nb_of_channels - 1) / 2 = nb_of_pairs
    Args:
        SCM_YY (np.ndarray):
            The spatial covariance matrix of the measured signal [nb_of_bins, nb_of_channels, nb_of_channels]
        SCM_VV (np.ndarray):
            The spatial covariance matrix of the noise [nb_of_bins, nb_of_channels, nb_of_channels]
        W (np.ndarray):
            Beamformer matrix [nb_of_doas, nb_of_pairs * nb_of_bins].
    Returns:
        P_mvdr (np.ndarray):
            The total pseudo-spectrum combining all frequency bins [nb_of_doas]
        P_mvdr_freq (np.ndarray):
            (optional, add manualy) The pseudo-spectrum at each frequency bin [nb_of_doas, nb_of_bins].
    """
    start = time.time()
    nb_of_bins, nb_of_channels, _ = SCM_YY.shape
    # Diagonal loading for inversion
    SCM_VV_loaded = shrinkage_diagonal_loading(SCM_VV, shrinkage_intensity=0.01) # [nb_of_channels, nb_of_channels, nb_of_bins]

    # Compute the inverse of SCM_VV_loaded for all frequency bins
    SCM_VV_inv = np.linalg.inv(SCM_VV_loaded)  # [nb_of_bins, nb_of_channels, nb_of_channels]

    # Compute the product of SCM_VV_inv and SCM_YY for all frequency bins
    Mat = np.einsum('ijk,ikl->ijl', SCM_VV_inv, SCM_YY)  # [nb_of_bins, nb_of_channels, nb_of_channels]
    
    # Get the indices for the upper triangular part, excluding the diagonal
    triu_indices = np.triu_indices(nb_of_channels, k=1)
    
    # Extract the upper triangular part of Mat for all frequency bins
    Mat_upper = Mat[:, triu_indices[0], triu_indices[1]]  # [nb_of_bins, nb_of_pairs]

    # Reshape Mat_upper into (nb_of_pairs * nb_of_bins, 1)
    Mat_reshaped = Mat_upper.T.flatten()[:, np.newaxis]  # [nb_of_pairs * nb_of_bins, 1]

    P_mvdr = np.real(W@Mat_reshaped)
        
    end_time = time.time()
    print('total calculatoin for MVDR pairwise',end_time-start)

    return P_mvdr.squeeze()

def beamformer_matrix_offline_calculation(TDOAs_scan, f):
    """
    Precalculate the beamformer matrix offline for improved online calculation time.

    Args:
        TDOAs_Scan (np.ndarray):
            The TDOAs in seconds [nb_of_channels, nb_of_doas]
        f (np.ndarray):
            Frequency bins vector [nb_of_bins,]
    Returns:
        W (np.ndarray):
            Beamformer matrix [nb_of_doas, nb_of_channels * (nb_of_channels - 1) / 2 ].
    """
    nb_of_channels, nb_of_doas = TDOAs_scan.shape
    nb_of_bins = f.shape[0]
    nb_of_pairs = (nb_of_channels*(nb_of_channels-1))//2
    
    # Beamformer matrix W calculation : can be precalculated offline
    TDOAs_scan = np.tile(TDOAs_scan[:,:,np.newaxis], (1, 1, nb_of_bins)) # [nb_of_channels, nb_of_doas, nb_of_bins]
    omega = 2*np.pi*f # [nb_of_bins,]
    
    W = np.zeros((nb_of_doas, nb_of_pairs*nb_of_bins), np.cdouble)
    pair_idx = 0
    for chan_id1 in range(nb_of_channels):
        # P_scan_1 = e^(jomega tau_i)
        Wi = np.exp(1j*omega*TDOAs_scan[chan_id1,:,:]) # [nb_of_doas, nb_of_bins]
        for chan_id2 in range(chan_id1+1, nb_of_channels):
            # Wj = e^(jomega tau_j)
            Wj = np.exp(1j*omega*TDOAs_scan[chan_id2,:,:]) # [nb_of_doas, nb_of_bins]

            # calculate beamformer matrix Wij [nb_of_doas, (N/2+1)]
            # Wij = e^(jomega (tau_i - tau_j))
            Wij = Wi*np.conj(Wj) # [nb_of_doas, nb_of_bins]

            # W = [W1-2, W1-3, ... W15-16 ] = [[nb_of_doas, nb_of_bins], [nb_of_doas, nb_of_bins],...,[nb_of_doas, nb_of_bins]]
            W[:, pair_idx * nb_of_bins:(pair_idx + 1) * nb_of_bins] = Wij
            pair_idx += 1
            
    return W
            
def shrinkage_diagonal_loading(SCMs, shrinkage_intensity=0.01, eps=1e-6):
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


