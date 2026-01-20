import numpy as np

def outlier_mask(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_limit = q1 - k * iqr
    upper_limit = q3 + k * iqr

    return (data >= lower_limit) & (data <= upper_limit)


def calc_velocity_sigmas_from_v_err(data, throw_id):
    """
    Estimate velocity uncertainty from v_err columns using RMS.

    If throw_id is given: compute RMS only on that throw.
    Otherwise: compute RMS over all throws.

    Returns:
      sigma_vx, sigma_vy, sigma_vz, sigma_vxy
    """

    dataset = data.copy()

    dataset = dataset[dataset['throw_id'] == throw_id].copy()

    # ignore first frames and NaNs
    dataset = dataset[(dataset['t'] > 0.0)].copy()

    ex = dataset['v_err_x'].dropna().values
    ey = dataset['v_err_y'].dropna().values
    ez = dataset['v_err_z'].dropna().values

    # RMS = sqrt(mean(err^2))
    sigma_vx = np.sqrt(np.mean(ex**2)) if len(ex) > 0 else np.nan
    sigma_vy = np.sqrt(np.mean(ey**2)) if len(ey) > 0 else np.nan
    sigma_vz = np.sqrt(np.mean(ez**2)) if len(ez) > 0 else np.nan

    # Combine horizontal uncertainty into one scalar for circles
    # (circle radius uses a single sigma value)
    sigma_vxy = np.sqrt((sigma_vx**2 + sigma_vy**2) / 2.0) if np.isfinite(sigma_vx) and np.isfinite(sigma_vy) else np.nan

    return sigma_vx, sigma_vy, sigma_vz, sigma_vxy