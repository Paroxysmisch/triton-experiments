import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE_D': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE_D': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE_D': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE_D': 4096}, num_warps=32),
    ],
    key=['N', 'D', 'B']
)
@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,  # Pointer to the input tensor
    z_ptr,  # Pointer to the output tensor
    N,      # Number of elements in the first dimension
    D,      # Number of elements in the last dimension
    B,      # Block size
    HAS_SCALE: tl.constexpr,  # Whether to scale the input
    SCALE: tl.float32,  # Scaling factor
    BLOCK_SIZE_D: tl.constexpr  # Block size for the last dimension
):
    pid_n = tl.program_id(axis=0)  # Block ID in the first dimension
    pid_d = tl.program_id(axis=1)  # Block ID in the last dimension

    # Compute the starting indices for the block
    i_n = pid_n
    i_d = pid_d * BLOCK_SIZE_D

    # Compute the range of indices to operate on
    o_d = i_d + tl.arange(0, BLOCK_SIZE_D)
    m_d = o_d < D

    # Load the input block, potentially scaling it
    b_x = tl.load(x_ptr + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * SCALE

    # Compute the maximum value in the block
    b_m = tl.max(b_x, axis=0)

    # Compute the log-sum-exp
    b_x = tl.exp(b_x - b_m)
    b_s = tl.sum(b_x, axis=0)
    b_z = tl.log(b_s) + b_m

    # Store the result
    tl.store(z_ptr + i_n * (D // B) + pid_d, b_z)

import torch
import triton
import triton.language as tl

def logsumexp_fwd(x, B=128, output_dtype=None):
    N, D = x.shape
    ND = D // B

    # Reshape the input tensor if necessary
    x = x.view(N, D)

    # Create an empty tensor to hold the results
    z = torch.empty((N, ND), device=x.device, dtype=x.dtype)

    # Determine the grid and block sizes
    grid = (N, ND)

    # Determine whether to scale the input
    HAS_SCALE = False
    SCALE = 1.0

    # Invoke the kernel
    logsumexp_fwd_kernel[grid](
        x, z, N, D, B, HAS_SCALE, SCALE, BLOCK_SIZE_D=B
    )

    # Reduce along the last dimension to produce the final result
    z = z.view(N, ND * B)
    z = torch.logsumexp(z, dim=-1)

    # Cast to the desired output data type if specified
    if output_dtype is not None:
        z = z.to(output_dtype)

    return z
