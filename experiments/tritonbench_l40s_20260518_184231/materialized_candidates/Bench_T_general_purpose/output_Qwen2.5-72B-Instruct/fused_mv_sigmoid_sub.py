import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, output_ptr, 
    n, m, alpha, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M

    # Compute the offsets for the current block
    offsets_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)

    # Load the input matrix and vector
    input_block = tl.load(input_ptr + offsets_m[:, None] * m + offsets_n[None, :], mask=offsets_m[:, None] < n, other=0.0)
    vec_block = tl.load(vec_ptr + offsets_n, mask=offsets_n < m, other=0.0)

    # Perform matrix-vector multiplication
    z_block = tl.dot(input_block, vec_block)

    # Apply sigmoid activation
    s_block = 1.0 / (1.0 + tl.exp(-z_block))

    # Load the other value
    other_block = tl.load(other_ptr + offsets_m, mask=offsets_m < n, other=0.0)

    # Compute the final output
    y_block = s_block - alpha * other_block

    # Store the result
    tl.store(output_ptr + offsets_m, y_block, mask=offsets_m < n)

import torch
import triton
import triton.language as tl

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Ensure input and vec are compatible for matrix-vector multiplication
    n, m = input.shape
    assert vec.shape == (m,), "vec must have shape (m,)"
    
    # Ensure other is broadcastable to the output shape (n,)
    if isinstance(other, torch.Tensor):
        assert other.shape == () or other.shape == (n,), "other must be a scalar or have shape (n,)"

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((n,), dtype=input.dtype, device=input.device)

    # Define grid and block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (triton.cdiv(n, BLOCK_SIZE_M),)

    # Launch the Triton kernel
    fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m, alpha,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )

    return out
