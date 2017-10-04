import numpy as np;
def rolling_window(arr, window):
    """Very basic multi dimensional rolling window. window should be the shape of
    of the desired subarrays. Window is either a scalar or a tuple of same size
    as `arr.shape`.
    """
    shape = np.array(arr.shape*2)
    strides = np.array(arr.strides*2)
    window = np.asarray(window)
    shape[arr.ndim:] = window # new dimensions size
    shape[:arr.ndim] -= window - 1
    if np.any(shape < 1):
        raise ValueError('window size is too large')
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)



def imageDistanceMS(arr, arr2):
    d = arr - arr2;
    pixelSize = float(arr.shape[0]*arr.shape[1]);
    rmsError = np.sum(np.square(d))/ pixelSize;
    return rmsError;



