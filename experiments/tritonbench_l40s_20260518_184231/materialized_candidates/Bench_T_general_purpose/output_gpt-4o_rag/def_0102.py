import triton
import triton.language as tl
import torch

# Triton kernel for softmax and multiplication
@triton.jit
def _softmax_mul_kernel(X, OTHER, OUT, x_stride, other_stride, out_stride, N, BLOCK_N: tl.constexpr, dim: tl.constexpr):
    # Compute program ID
    pid = tl.program_id(axis=0)
    # Create pointers for the input and output
    x_ptrs = X + pid * x_stride
    other_ptrs = OTHER + pid * other_stride
    out_ptrs = OUT + pid * out_stride
    
    # Load the data
    x = tl.load(x_ptrs + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=-float('inf'))
    # Compute the maximum for numerical stability
    max_x = tl.max(x, axis=0)
    # Subtract max and exponentiate
    x = tl.exp(x - max_x)
    # Sum the exponentials
    sum_x = tl.sum(x, axis=0)
    # Normalize to get softmax
    softmax_x = x / sum_x
    # Load other and multiply
    other = tl.load(other_ptrs + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=1.0)
    result = softmax_x * other
    # Store the result
    tl.store(out_ptrs + tl.arange(0, BLOCK_N), result, mask=tl.arange(0, BLOCK_N) < N)

# Wrapper function for softmax_mul
def softmax_mul(input, other, dim, dtype=None, out=None):
    assert input.is_cuda, "Input must be a CUDA tensor"
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"

    # Determine the size of the dimension along which to apply softmax
    N = input.size(dim)
    BLOCK_N = 1024  # This is a tuning parameter that can be adjusted
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    # Call the Triton kernel
    _softmax_mul_kernel[(num_blocks,)](
        input, other, out,
        input.stride(dim), other.stride(dim), out.stride(dim),
        N, BLOCK_N=BLOCK_N, dim=dim
    )

    return out
