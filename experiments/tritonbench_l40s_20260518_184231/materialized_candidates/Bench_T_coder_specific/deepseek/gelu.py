import triton
import numpy as np

def gelu(input, approximate='none'):
    if approximate == 'none':
        return input * triton.sigmoid(input * np.sqrt(2 / np.pi))
    elif approximate == 'tanh':
        return 0.5 * input * (1 + np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * input**3)))
    else:
        raise ValueError("Approximation method must be either 'none' or 'tanh'")
