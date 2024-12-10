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

def create_maps(
    t,                 # Time vector or index
    f=None,            # Signal vector (optional; defaults to `t`)
    maxtau=10,         # Maximum value for tau
    mfunc="mean",          # Method for central tendency (default: "mean")
    vfunc="volume",           # Method for variability (default: "var")
    inter_method="linear"  # Interpolation method (default: "linear")
):
    """
    Generates Variability and Mean maps (Vmap and Mmap) for a given signal.
    """
    # If no signal vector is provided, use `t` as the signal and generate indices for `t`
    if f is None:
        f = t
        t = np.arange(1, len(f) + 1)  # Create indices starting from 1 (like R)

    # Initialize Vmap and Mmap matrices (filled with zeros)
    VM = np.zeros((len(f), maxtau))
    MM = np.zeros((len(f), maxtau))

    # Compute Vmap and Mmap for each value of tau
    for tau in range(2, maxtau + 1):  # Start tau from 2
        out = ebt(t, f, tau=tau, mfunc=mfunc, vfunc=vfunc, inter_method=inter_method)  # Call ebt function
        VM[:, tau - 1] = out["V"]  # Populate Variability map
        MM[:, tau - 1] = out["M"]  # Populate Mean map

    # Return the results as a dictionary
    return {
        "t": t,      # Time vector
        "vmap": VM,  # Variability map
        "mmap": MM   #
     }

def extract_signal(
    time_vector,         # The time vector of the input data
    signal,              # The signal vector of the input data
    tau,                 # Parameter for signal filtering
    mfunc="mean",        # Method for estimating the central tendency (default: "mean")
    vfunc="var",         # Method for estimating variability (default: "var")
    tol=0.05,            # Convergence threshold (stopping condition for iterations)
    maxiter=100,         # Maximum number of iterations
    inter_method="linear" # Method for interpolation (default: "linear")
):
    """
    Iteratively extracts the residual signal by applying the EBT method.

    Parameters:
    -----------
    time_vector : array-like
        The time vector of the input data.
    signal : array-like
        The signal vector of the input data.
    tau : int
        Parameter for signal filtering.
    mfunc : str
        Method for central tendency estimation (default: "mean").
    vfunc : str
        Method for variability estimation (default: "var").
    tol : float
        Convergence threshold for stopping condition (default: 0.05).
    maxiter : int
        Maximum number of iterations (default: 100).
    inter_method : str
        Interpolation method used in EBT (default: "linear").

    Returns:
    --------
    dict :
        A dictionary containing:
        - "final_residual_signal": The residual signal after convergence.
        - "converged": Whether the algorithm converged within the maximum iterations.
        - "iterations": The number of iterations performed.
    """
    # Initialize the filtered signal and residual
    prev_filtered_signal = np.array(signal, dtype=np.float64)
    iteration = 0  # Track the number of iterations

    # Apply EBT filtering for the first iteration
    ebt_output = ebt(
        time_vector, signal, tau=tau, mfunc=mfunc, vfunc=vfunc, inter_method=inter_method
    )
    current_filtered_signal = ebt_output['M']
    residual_signal = signal - current_filtered_signal

    # Iterative refinement process
    while (
        np.max(np.abs(prev_filtered_signal - current_filtered_signal)) > tol
        and iteration < maxiter
    ):
        # Update the filtered signal
        prev_filtered_signal = current_filtered_signal

        # Apply EBT to the residual signal
        ebt_output = ebt(
            time_vector, residual_signal, tau=tau, mfunc=mfunc, vfunc=vfunc, inter_method=inter_method
        )
        current_filtered_signal = ebt_output['M']

        # Update the residual signal
        residual_signal = residual_signal - current_filtered_signal

        # Increment iteration counter
        iteration += 1

    # Check convergence status
    converged = iteration < maxiter

    # Return the final residual signal and metadata
    return residual_signal