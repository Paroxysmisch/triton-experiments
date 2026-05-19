import triton
import numpy as np

def fused_hardshrink_dropout(input: np.ndarray, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> np.ndarray:
    if not training:
        return input

    output = np.empty_like(input)
    grid = lambda x: (x + 255) // 256 * 256
    block = 256
    fused_hardshrink_dropout_kernel[grid(output.size), block](output, input, p, inplace, output.size, lambd)

    return output
