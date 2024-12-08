import numpy as np
from scipy.interpolate import CubicSpline, interp1d

def ebt(t, f=None, tau=1, mfunc="mean", vfunc="var", inter_method="linear"):
    """
    EBT (Empirical Boundary Transformation) implementation in Python.
    
    Parameters:
    - t: Time vector
    - f: Signal vector (optional; defaults to `t`)
    - tau: Parameter for filtering
    - mfunc: Method for calculating central tendency ("mean", "median", or "volume")
    - vfunc: Method for calculating variability ("var" or "volume")
    - inter_method: Interpolation method ("linear" or "cubic")
    
    Returns:
    A dictionary with keys:
    - t: Original time vector
    - f: Original signal vector
    - V: Variability vector
    - L: Lower bounds
    - U: Upper bounds
    - M: Central tendency vector
    - tau: Parameter tau
    - band: Interpolated band matrix
    - knot: Sampled indices
    """
    
    # If no signal vector is provided, set `f` as `t` and generate `t` as an index
    if f is None:
        f = t
        t = np.arange(1, len(f) + 1)
    
    # Save original time and signal vectors
    tsave = t.copy()
    fsave = f.copy()
    
    # Extend signal with mirrored values for boundary handling
    f = np.concatenate([np.full(tau * 2, f[0]), f, np.full(tau * 2, f[-1])])
    t = np.arange(1, len(f) + 1)
    length = len(f)
    
    # Initialize variables
    sampled_index = []
    missing_index = []
    band = np.zeros((length, tau))
    
    # Perform interpolation based on the specified method
    for eta in range(tau):
        sampled_idx = np.arange(eta, length, tau)
        
        if sampled_idx[0] != 0:
            sampled_idx = np.insert(sampled_idx, 0, 0)
        
        if sampled_idx[-1] != length - 1:
            sampled_idx = np.append(sampled_idx, length - 1)
        
        sampled_index.append(sampled_idx)
        missing_idx = np.setdiff1d(np.arange(length), sampled_idx)
        missing_index.append(missing_idx)
        
        if inter_method == "cubic":
            cubic_spline = CubicSpline(t[sampled_idx], f[sampled_idx])
            band[sampled_idx, eta] = cubic_spline(t[sampled_idx])
            band[missing_idx, eta] = cubic_spline(t[missing_idx])
        elif inter_method == "linear":
            linear_interp = interp1d(t[sampled_idx], f[sampled_idx], kind='linear', fill_value="extrapolate")
            band[:, eta] = linear_interp(t)
    
    # Compute upper and lower bounds
    U = np.max(band, axis=1)
    L = np.min(band, axis=1)
    
    # Compute mean, median, and volume-based central tendency
    M1 = np.mean(band, axis=1)
    M2 = np.median(band, axis=1)
    M3 = (L + U) / 2
    
    # Compute variability metrics
    V1 = U - L
    V2 = np.var(band, axis=1, ddof=0) * (tau - 1) / tau
    
    # Select central tendency based on `mfunc`
    if mfunc == "mean":
        M = M1
    elif mfunc == "median":
        M = M2
    elif mfunc == "volume":
        M = M3
    
    # Select variability metrics based on `vfunc`
    if vfunc == "volume":
        V = V1
    elif vfunc == "var":
        V = V2
        L = M - V
        U = M + V
    
    # Define index for the output
    index = np.arange(2 * tau, len(f) - 2 * tau)
    
    # Filter sampled indices to match the length of the original signal
    filtered_sampled_index = [s[s < len(fsave)] for s in sampled_index]
    
    # Return results as a dictionary
    return {
        't': tsave,               # Original time vector
        'f': fsave,               # Original signal vector
        'V': V[index],            # Variability vector
        'L': L[index],            # Lower bounds
        'U': U[index],            # Upper bounds
        'M': M[index],            # Central tendency vector
        'tau': tau,               # Parameter tau
        'band': band[index, :],   # Interpolated band matrix
        'knot': filtered_sampled_index # Sampled indices
    }

