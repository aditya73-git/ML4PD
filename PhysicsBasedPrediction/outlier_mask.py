import numpy as np

def outlier_mask(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_limit = q1 - k * iqr
    upper_limit = q3 + k * iqr

    return (data >= lower_limit) & (data <= upper_limit)