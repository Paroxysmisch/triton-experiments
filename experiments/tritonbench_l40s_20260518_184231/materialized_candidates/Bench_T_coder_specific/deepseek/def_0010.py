import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

def linalg_svd(A, full_matrices=True, driver=None):
    if A.dtype not in (dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128):
        raise ValueError("A must be of type float32, float64, complex64, or complex128")

    if A.shape.ndims < 2:
        raise ValueError("A must be at least a 2D tensor")

    if driver not in (None, 'gesvd', 'gesvdj', 'gesvda'):
        raise ValueError("Invalid driver. Available options are: None, 'gesvd', 'gesvdj', and 'gesvda'")

    svd_out = tf.linalg.svd(A, full_matrices=full_matrices, compute_uv=True)

    return svd_out
