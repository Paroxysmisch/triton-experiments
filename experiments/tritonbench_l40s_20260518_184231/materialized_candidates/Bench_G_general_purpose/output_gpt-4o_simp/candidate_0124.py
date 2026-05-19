import triton
import triton.language as tl
import torch

@triton.jit
def logsumexp_fwd_kernel(
    x_ptr, z_ptr, scale_ptr,
    D, B,
    BLOCK_SIZE: tl.constexpr,
    ):
    # Get the program ID for the current block
    pid = tl.program_id(0)

    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE

    # Create a pointer for the input block
    x_block_ptr = x_ptr + block_start

    # Load the input block
    x = tl.load(x_block_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < D, other=-float('inf'))

    # Optionally apply the scale
    if scale_ptr is not None:
        scale = tl.load(scale_ptr)
        x *= scale

    # Compute the maximum value in the block
    max_x = tl.max(x, axis=0)

    # Compute the exponential and sum it
    exp_x = tl.exp(x - max_x)
    sum_exp_x = tl.sum(exp_x, axis=0)

    # Compute the log-sum-exp
    log_sum_exp = max_x + tl.log(sum_exp_x)

    # Store the result in the output tensor
    z_block_ptr = z_ptr + pid
    tl.store(z_block_ptr, log_sum_exp)


def logsumexp_fwd(x, scale=None):
    # Get the dimensions of the input tensor
    shape = x.shape
    D = shape[-1]
    num_blocks = (D + 1024 - 1) // 1024  # Compute the number of blocks needed
    B = min(D, 1024)  # Set block size, typically a power of two

    # Create the output tensor
    z = torch.empty(shape[:-1], dtype=x.dtype, device=x.device)

    # Launch the Triton kernel
    grid = (num_blocks,)
    logsumexp_fwd_kernel[grid](
        x_ptr=x,
        z_ptr=z,
        scale_ptr=scale,
        D=D,
        B=B,
        BLOCK_SIZE=B
    )

    return z
