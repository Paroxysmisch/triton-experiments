import triton
import triton.language as tl
import torch
from torch import Tensor

# Triton kernel for softmax multiplication
@triton.jit
def _softmax_mul(input, other, OUT, input_stride, other_stride, out_stride, dim, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    # Initialize output
    out = tl.zeros((BLOCK_N,), tl.float32)
    
    # Load input and other tensors
    input_ptr = input + rm * input_stride
    other_ptr = other + rm * other_stride
    
    # Compute softmax
    x = tl.load(input_ptr, mask=tl.arange(0, BLOCK_N) < N)
    max_x = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_x)
    softmax_x = exp_x / tl.sum(exp_x, axis=0)
    
    # Multiply by other tensor or number
    other_val = tl.load(other_ptr, mask=tl.arange(0, BLOCK_N) < N)
    out = softmax_x * other_val
    
    # Store result
    out_ptr = OUT + rm * out_stride
    tl.store(out_ptr, out)

# Wrapper function for softmax_mul
def softmax_mul(input: Tensor, other: Tensor, dim: int, dtype=None, out: Tensor = None) -> Tensor:
    assert input.is_cuda and (other.is_cuda or isinstance(other, (int, float)))
    
    # Cast input to desired dtype if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Prepare output tensor
    if out is None:
        out = input.new_empty(input.shape)
    
    # Get dimensions
    *dims, N = input.shape
    input_stride = input.stride(dim)
    other_stride = other.stride(0) if isinstance(other, Tensor) else 0
    out_stride = out.stride(0)
    
    # Call the Triton kernel
    _softmax_mul[(input.shape[0],)](input, other, out, input_stride, other_stride, out_stride, dim, N, BLOCK_N=1024)
    
    return out
