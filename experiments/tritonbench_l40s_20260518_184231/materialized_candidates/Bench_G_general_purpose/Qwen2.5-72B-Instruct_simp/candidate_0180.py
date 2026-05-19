import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load the relevant section of X
    x_ptr = X + row * stride_x_row + cols
    x = tl.load(x_ptr, mask=mask, other=0.0)

    # Compute the L2 norm
    x_squared = x * x
    sum_x_squared = tl.sum(x_squared, axis=0)
    norm = tl.sqrt(sum_x_squared + eps)

    # Normalize the data
    y = x / norm

    # Store the result in Y
    y_ptr = Y + row * stride_x_row + cols
    tl.store(y_ptr, y, mask=mask)

import torch
import triton
import triton.language as tl

def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-6, BLOCK_N: int = 1024):
    # Ensure x is contiguous
    if not x.is_contiguous():
        x = x.contiguous()

    # Prepare output tensor
    y = torch.empty_like(x)

    # Check constraints on the maximum allowable block size
    if BLOCK_N > triton.runtime.max_block_size():
        raise ValueError(f"Block size {BLOCK_N} exceeds the maximum allowable block size {triton.runtime.max_block_size()}")

    # Launch the Triton kernel
    grid = (x.shape[0],)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y, x.stride(1), x.shape[1], eps, BLOCK_N
    )

    return y
