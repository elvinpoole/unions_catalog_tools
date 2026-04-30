import numpy as np
import h5py
import pandas as pd

def compute_des_weights(
    cat_path,
    num_bins,
    snr_col='snr',
    size_ratio_col=None,   # if None, computed from NGMIX_T_NOSHEAR / NGMIX_Tpsf_NOSHEAR
    snr_min=None,
    snr_max=None,
    size_ratio_min=None,
    size_ratio_max=None,
    step=0.01,
    sign=1.0,
):
    """
    Compute DES-style weights (Gatti et al. 2021) from a cut catalog HDF5.
    Bins galaxies in log(SNR) x log(size_ratio) and assigns each object a
    weight w = R^2 / sigma_e^2, where R is the mean scalar shear response
    and sigma_e^2 is the shape noise in that bin.

    Parameters
    ----------
    cat_path       : path to HDF5 cut catalog
    num_bins       : number of bins along each axis
    snr_col        : column name for SNR (default: 'snr')
    size_ratio_col : column name for size ratio. If None, computed as
                     NGMIX_T_NOSHEAR / NGMIX_Tpsf_NOSHEAR
    snr_min/max    : SNR bin edges (None = use data min/max)
    size_ratio_min/max : size ratio bin edges (None = use data min/max)
    step           : shear step used to compute R columns
    sign           : 1.0 for ShapePipe, -1.0 for GALSIM

    Returns
    -------
    w_des : np.ndarray, shape (N,)
    """
    with h5py.File(cat_path, 'r') as f:
        e1  = f['e1_uncal'][:]
        e2  = f['e2_uncal'][:]
        R11 = f['R11'][:]
        R22 = f['R22'][:]
        snr = f[snr_col][:]

        if size_ratio_col is not None:
            size_ratio = f[size_ratio_col][:]
        else:
            # Derive size ratio from T columns present in the catalog
            T_noshear   = f['NGMIX_T_NOSHEAR'][:]
            Tpsf_noshear = f['NGMIX_Tpsf_NOSHEAR'][:]
            size_ratio  = T_noshear / Tpsf_noshear

    # --- build a flat DataFrame for easy bin indexing ---
    df = pd.DataFrame({
        'e1':         e1,
        'e2':         e2,
        'R11':        R11,
        'R22':        R22,
        'snr':        snr,
        'size_ratio': size_ratio,
    })

    # Drop rows with non-finite values (NaN/Inf from division or bad measurements)
    n_raw = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n_dropped = n_raw - len(df)
    if n_dropped > 0:
        print(f"Warning: dropped {n_dropped} objects with non-finite values before binning.")

    # --- log-spaced bin edges ---
    snr_lo = snr_min        if snr_min        is not None else df['snr'].min()
    snr_hi = snr_max        if snr_max        is not None else df['snr'].max()
    sr_lo  = size_ratio_min if size_ratio_min is not None else size_ratio.min()
    sr_hi  = size_ratio_max if size_ratio_max is not None else size_ratio.max()

    snr_edges = np.logspace(np.log10(snr_lo), np.log10(snr_hi), num_bins + 1)
    sr_edges  = np.logspace(np.log10(sr_lo),  np.log10(sr_hi),  num_bins + 1)

    # assign each object to a bin (0-indexed; clip to valid range)
    df['snr_bin'] = (np.searchsorted(snr_edges, df['snr'].values,        side='right') - 1).clip(0, num_bins - 1)
    df['sr_bin']  = (np.searchsorted(sr_edges,  size_ratio, side='right') - 1).clip(0, num_bins - 1)

    df['w_des'] = np.nan

    # --- fill weights bin by bin ---
    for i in range(num_bins):
        for j in range(num_bins):
            mask = (df['snr_bin'] == i) & (df['sr_bin'] == j)
            ngal = mask.sum()
            if ngal == 0:
                continue

            shape_noise = 0.5 * (
                (df.loc[mask, 'e1'] ** 2).mean()
                + (df.loc[mask, 'e2'] ** 2).mean()
            )

            # scalar response: mean of diagonal elements
            response = 0.5 * (
                df.loc[mask, 'R11'].mean()
                + df.loc[mask, 'R22'].mean()
            )

            df.loc[mask, 'w_des'] = response ** 2 / shape_noise if shape_noise > 0 else 0.0

    # --- reconstruct full-length output (dropped rows get weight 0) ---
    w_out = np.zeros(n_raw, dtype=np.float64)
    w_out[df.index] = df['w_des'].fillna(0.0).values

    n_missing = (w_out == 0).sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} objects have weight 0 (empty bins or non-finite inputs).")

    return w_out

def compute_total_response(cat_path, step=0.01, sign=1.0, weight_col='w_iv'):
    """
    Compute shear + selection response from a cut catalog HDF5 file.
    Parameters
    ----------
    cat_path   : path to the HDF5 cut catalog
    step       : shear step size used when generating 1P/1M/2P/2M columns
    sign       : 1.0 for ShapePipe, -1.0 for GALSIM
    weight_col : column name (str), pre-computed weight array (np.ndarray),
                 or None for unweighted
    Returns
    -------
    R_shear     : 2x2 array, per-object shear response averaged over catalog
    R_selection : 2x2 array, selection response
    R_total     : 2x2 array, R_shear + R_selection
    """
    h2 = 2 * step

    with h5py.File(cat_path, 'r') as f:
        R11 = f['R11'][:]
        R22 = f['R22'][:]
        R12 = f['R12'][:]
        R21 = f['R21'][:]
        g1_p1 = f['NGMIX_ELL_1P_0'][:]
        g2_p1 = f['NGMIX_ELL_1P_1'][:]
        g1_m1 = f['NGMIX_ELL_1M_0'][:]
        g2_m1 = f['NGMIX_ELL_1M_1'][:]
        g1_p2 = f['NGMIX_ELL_2P_0'][:]
        g2_p2 = f['NGMIX_ELL_2P_1'][:]
        g1_m2 = f['NGMIX_ELL_2M_0'][:]
        g2_m2 = f['NGMIX_ELL_2M_1'][:]

        # load weights from file only if a column name was given
        if isinstance(weight_col, str):
            weights = f[weight_col][:]
        elif isinstance(weight_col, np.ndarray):
            weights = weight_col
        else:
            weights = None

    # --- Shear response: weighted mean of per-object R matrix ---
    if weights is not None:
        R11_mean = np.average(R11, weights=weights)
        R22_mean = np.average(R22, weights=weights)
        R12_mean = np.average(R12, weights=weights)
        R21_mean = np.average(R21, weights=weights)
    else:
        R11_mean = np.mean(R11)
        R22_mean = np.mean(R22)
        R12_mean = np.mean(R12)
        R21_mean = np.mean(R21)

    R_shear = np.array([[R11_mean, R12_mean],
                         [R21_mean, R22_mean]])

    R11_s = (np.mean(g1_p1) - np.mean(g1_m1)) / h2
    R22_s = sign * (np.mean(g2_p2) - np.mean(g2_m2)) / h2
    R12_s = (np.mean(g1_p2) - np.mean(g1_m2)) / h2
    R21_s = (np.mean(g2_p1) - np.mean(g2_m1)) / h2

    R_selection = np.array([[R11_s, R12_s],
                              [R21_s, R22_s]])

    R_total = R_shear + R_selection
    return R_shear, R_selection, R_total

# --- Run it ---

cat_path = 'output_mpi_wTcut_ugriz_cutcat/unions_shapepipe_cutcat_ugriz_2024_v1.6.c.1.hdf5'

# --- example usage ---
w_des = compute_des_weights(
    cat_path=cat_path,
    num_bins=20,
    snr_col='snr',
    size_ratio_col=None,   # you'll need to add these to cutcat_cols
    snr_min=10,
    snr_max=500,
    size_ratio_min=0.707,
    size_ratio_max=3.0,
)

print(f"w_des: min={w_des.min():.4f}, max={w_des.max():.4f}, mean={w_des.mean():.4f}")


R_shear, R_sel, R_total = compute_total_response(
    cat_path,
    step=0.01,
    sign=1.0,
    weight_col='w_iv',
)

print("R_shear:")
print(R_shear)
print("\nR_selection:")
print(R_sel)
print("\nR_total:")
print(R_total)
print(f"\nR_total diagonal: R11={R_total[0,0]:.4f}, R22={R_total[1,1]:.4f}")
print(f"Scalar response (mean of diagonal): {0.5*(R_total[0,0]+R_total[1,1]):.4f}")