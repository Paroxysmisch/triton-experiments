import triton
import triton.language as tl

@triton.jit
def _abs_max(val1, val2):
    # Calculate the absolute maximum of two values.
    val1_abs = tl.abs(val1)
    val2_abs = tl.abs(val2)
    if val1_abs >= val2_abs:
        return val1_abs
    else:
        return val2_abs
