import numpy as np
from scipy.io import wavfile

def get_signal(abs_path):
    fs, sig = wavfile.read(abs_path)
    T = sig.shape[0]/fs
    return fs, sig, T


def sig_same_size(sig1, sig2):

    if sig1.ndim == 1:
        max_size = np.min([sig1.shape[0], sig2.shape[0]])

        if sig1.shape[0] > max_size:
            sig1 = sig1[:max_size]

        if sig2.shape[0] > max_size:
            sig2 = sig2[:max_size]

    elif sig1.ndim == 2:
        max_size = np.min([sig1.shape[1], sig2.shape[1]])

        if sig1.shape[1] > max_size:
            sig1 = sig1[:,:max_size]

        if sig2.shape[1] > max_size:
            sig2 = sig2[:,:max_size]

    return sig1, sig2


def max_rolling(A,K):
    if A.ndim != 4:  # only works for ndim = 4
          return A
    
    frame_size = A.shape[-1]

    # Shift cross-correlation to 0 delay = center
    A_new = np.zeros(A.shape)
    A_new[:,:,:,:frame_size//2] = A[:,:,:,frame_size//2:]
    A_new[:,:,:,frame_size//2:] = A[:,:,:,:frame_size//2]

    # Rolling max window of K
    A_new2 = np.zeros(A_new.shape, dtype=np.float32)
    for jj in range(K//2, A.shape[-1] - K//2):
        A_new2[:,:,:,jj] = np.max(A_new[:,:,:,jj-K//2:jj+K//2+1], axis=-1)

    # Reshift cross-correlation to 0 delay = [0] and [-1]
    A_rollingmax = np.zeros(A.shape)
    A_rollingmax[:,:,:,:frame_size//2] = A_new2[:,:,:,frame_size//2:]
    A_rollingmax[:,:,:,frame_size//2:] = A_new2[:,:,:,:frame_size//2]

    return A_rollingmax

