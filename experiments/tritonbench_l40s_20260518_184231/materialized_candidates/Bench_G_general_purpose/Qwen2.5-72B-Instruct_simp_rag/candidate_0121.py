import triton
import triton.language as tl
import torch

# Helper function to find the next power of 2
def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

# Helper function to determine the number of warps
def num_warps(n):
    if n < 512:
        return 4
    if n < 2048:
        return 8
    return 16

# Triton kernel for log-sum-exp forward pass
@triton.jit
def logsumexp_fwd_kernel(
    X, Z, scale, D, B, stride_x, stride_z, **meta
):
    pid = tl.program_id(0)
    block_size = meta['BLOCK']
    num_blocks = (D + block_size - 1) // block_size
    block_id = pid % num_blocks
    batch_id = pid // num_blocks

    # Compute the range of elements to process
    start = block_id * block_size
    end = min(start + block_size, D)

    # Initialize pointers
    x_ptr = X + batch_id * stride_x + start
    z_ptr = Z + batch_id * stride_z

    # Load the segment of the input tensor
    x = tl.load(x_ptr, mask=start + tl.arange(0, block_size) < D, other=-float('inf'))

    # Optionally scale the input
    if scale != 0.0:
        x = x * scale

    # Compute the maximum value
    max_val = tl.max(x, axis=0)

    # Compute the log-sum-exp
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_val

    # Store the result
    tl.store(z_ptr, log_sum_exp)

# Wrapper function for the log-sum-exp forward pass
def logsumexp_fwd(x, scale=1.0, dtype=None):
    # Reshape the input tensor
    shape = x.shape
    D = shape[-1]
    B = next_power_of_2(D)
    x = x.view(-1, D)

    # Determine the appropriate block size
    block_size = num_warps(D) * 32

    # Create an empty output tensor
    z = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)

    # Launch the kernel
    grid = lambda opt: (x.shape[0],)
    logsumexp_fwd_kernel[grid](x, z, scale, D, B, x.stride(0), z.stride(0), BLOCK=block_size)

    # Reshape the result back to the original shape minus the last dimension
    z = z.view(*shape[:-1])

    # Optionally cast the result to the specified data type
    if dtype is not None:
        z = z.to(dtype)

    return z
