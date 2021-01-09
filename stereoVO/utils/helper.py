import numpy as np
import math

def rmse_error(predicted, actual):
    """
    computes the root-mean-square between two vectors.
    :param predicted (np.array) of size(N)
    :param actual (np.array) of size(N)
    
    Returns
        rmse_error (float64)
    """
    return np.sqrt(np.mean((predicted-actual)**2))