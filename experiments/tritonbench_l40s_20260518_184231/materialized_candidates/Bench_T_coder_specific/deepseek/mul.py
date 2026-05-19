import numpy as np
import triton

def mul(input, other, out=None):
    # Convert inputs to NumPy arrays if they're not already
    if not isinstance(input, np.ndarray):
        input = np.array(input)
    if not isinstance(other, np.ndarray):
        other = np.array(other)

    # Perform the multiplication
    result = np.multiply(input, other, out)

    # If an output tensor was provided, return the result in it
    if out is not None:
        return out
    else:
        return result
