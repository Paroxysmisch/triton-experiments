import triton
from triton.language import *
import numpy as np

@triton.jit
def min_gelu_kernel(
    input_ptr: Pointer[Tensor],
    output_ptr: Pointer[Tensor],
    input_shape: Tensor[int64],
    reduction_dim: int64,
    keepdim: bool,
    approximate: str,
    num_elements: int64,
):
    idx = tl.program_id(axis=0)
    gelu_value = tl.zeros([], dtype=tl.float32)

    # Compute GELU based on the specified approximation method
    if approximate == "none":
        gelu_value = input_ptr[idx] * tl.erff(input_ptr[idx] / tl.sqrt(tl.float32(2.0)))
    elif approximate == "tanh":
        tanh_term = tl.tanh(tl.sqrt(tl.float32(2.0 / tl.math.pi)) * (input_ptr[idx] + 0.044715 * tl.pow(input_ptr[idx], 3)))
        gelu_value = 0.5 * input_ptr[idx] * (1.0 + tanh_term)
    else:
        gelu_value = input_ptr[idx]  # Fallback, should not happen

    # Reduce along the specified dimension
    if reduction_dim == -1:
        tl.atomic_min(output_ptr[0], gelu_value)
    else:
        reduced_idx = idx // input_shape[reduction_dim]
        output_ptr[reduced_idx] = tl.fmin(output_ptr[reduced_idx], gelu_value)

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    # Prepare input tensor
    input_tensor = input.contiguous()
    input_shape = input_tensor.shape
    num_elements = input_tensor.numel()

    # Create output tensor
    if out is None:
        if dim is None:
            out_shape = (1,)
        else:
            out_shape = tuple(size if i != dim else 1 for i, size in enumerate(input_shape))
        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(num_elements, meta.block_size),)
    block_size = 256
    min_gelu_kernel[grid](input_tensor.data_ptr(), out.data_ptr(), input_shape, dim, keepdim, approximate, num_elements, block=(block_size, 1, 1))

    return out
