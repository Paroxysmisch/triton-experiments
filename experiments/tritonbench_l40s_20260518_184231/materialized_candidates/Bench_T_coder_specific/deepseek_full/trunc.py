import triton
import triton.language as tl
from triton.language.libdevice import trunc as libdevice_trunc

@triton.jit
def trunc(x):
    # For integer inputs, follows the array-api convention of returning a copy of the input tensor.
    return libdevice_trunc(x) if x.dtype.is_floating() else tl.make_tensor(x.to_numpy())

def wrapper_trunc(input, *, out=None):
    return trunc(input, out=out)
