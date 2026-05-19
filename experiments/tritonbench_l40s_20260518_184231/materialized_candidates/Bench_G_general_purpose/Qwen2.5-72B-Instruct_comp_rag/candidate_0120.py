import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
    ],
    key=['N', 'D'],
)
@triton.jit
def logsumexp_fwd_kernel(
    x_ptr, z_ptr, N, D, BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Compute the block start indices
    i_n = pid_n * BLOCK_SIZE
    i_d = pid_d * BLOCK_SIZE

    # Compute the block end indices
    o_d = tl.min(i_d + BLOCK_SIZE, D)
    m_d = tl.min(i_d + BLOCK_SIZE, D)

    # Create a mask to avoid out-of-bounds memory accesses
    mask = i_d + tl.arange(0, BLOCK_SIZE) < m_d

    # Load the input block
    b_x = tl.load(x_ptr + i_n * D + i_d, mask=mask, other=-float('inf'))

    # Compute the maximum value in the block
    b_m = tl.max(b_x, axis=0)

    # Compute the log-sum-exp
    b_x = b_x - b_m
    b_sum = tl.sum(tl.exp(b_x), axis=0)
    b_logsumexp = tl.log(b_sum) + b_m

    # Store the result
    tl.store(z_ptr + i_n * D + i_d, b_logsumexp, mask=mask)

import torch

def logsumexp_fwd(x, output_dtype=None):
    N, D = x.shape
    B = 128  # Block size for the kernel
    ND = (D + B - 1) // B

    # Create an empty tensor to hold the results
    z = torch.empty((N, D), dtype=x.dtype, device=x.device)

    # Determine the grid size
    grid = (N, ND)

    # Invoke the kernel
    logsumexp_fwd_kernel[grid](x, z, N, D, BLOCK_SIZE=B)

    # Reduce along the last dimension to produce the final result
    z = z.logsumexp(dim=-1)

    # Cast to the desired output data type if specified
    if output_dtype is not None:
        z = z.to(output_dtype)

    return z
