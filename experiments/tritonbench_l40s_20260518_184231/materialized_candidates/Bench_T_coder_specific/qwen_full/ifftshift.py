import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor import triton_helpers

@triton.jit
def ifftshift_(x, stride, N, dim=0):
    # Calculate the shift amount, which is half the size of the dimension
    shift = tl.where(N % 2 == 0, N // 2, N // 2 - 1)
    # Create a range of indices for the dimension
    idx = tl.arange(0, N)
    # Apply the ifftshift logic
    return tl.where(idx < shift, idx + stride * (N - shift), idx - stride * (shift + 1))

@triton.jit
def ifftshift(x, stride, N, dim=0):
    # Determine if the dimension is negative and adjust if necessary
    if dim < 0:
        dim += len(N)
    # Calculate the new shape and strides
    new_shape = list(x.shape)
    new_strides = list(x.stride())
    new_shape[dim] = N
    new_strides[dim] = stride
    # Apply the ifftshift_ function and return the result
    return ifftshift_(x, stride, N, dim).view(new_shape, new_strides)

def ifftshift(input, dim=None):
    # Determine the input shape and strides
    shape = input.shape
    strides = input.stride()
    if dim is None:
        # Flatten the input and apply ifftshift
        input = input.reshape(-1)
        N = shape[-1]
        return ifftshift(input, strides[-1], N, -1).reshape(shape)
    else:
        # Ensure dim is a list for uniform processing
        dim = dim if isinstance(dim, list) else [dim]
        # Calculate the product of dimensions not being shifted
        prod = 1
        for i in range(len(shape)):
            if i not in dim:
                prod *= shape[i]
        # Apply ifftshift to each block of the specified dimension
        for d in dim:
            N = shape[d]
            orig = input.view(prod, shape[d], *shape[d + 1 :])
            shifted = ifftshift(orig, strides[d], N, 1)
            input = shifted.reshape(prod, *shape[d + 1 :])
        return input
