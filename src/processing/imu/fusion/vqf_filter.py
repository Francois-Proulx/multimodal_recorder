from vqf import VQF, offlineVQF
import numpy as np

def run_vqf(Ts, gyro, acc, mag=None, offline=False, params=None):
    """Run VQF sensor fusion and return quaternion array.
    Uses the Scalar-first convention (w, x, y, z).
    """
    if offline:
        out = offlineVQF(gyro, acc, mag, Ts, params)
        quat = out['quat9D'] if mag is not None else out['quat6D']
    else:
        vqf = VQF(Ts, **(params or {}))
        if mag is not None:
            out = vqf.updateBatch(gyro, acc, mag)
            quat = out['quat9D']
        else:
            out = vqf.updateBatch(gyro, acc)
            quat = out['quat6D']
    
    # Change from (w, x, y, z) to (x, y, z, w) for scipy compatibility
    quat = quat[:, [1, 2, 3, 0]]
    return quat