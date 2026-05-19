import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    X,  # pointer to the input tensor
    Y,  # pointer to the output tensor
    M,  # number of rows
    N,  # number of columns
    TILE_N: tl.constexpr,  # tile size, power of 2
):
    # Compute the block ID in the grid
    pid = tl.program_id(0)
    num_pid_m = M
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group // num_pid_m
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) * TILE_N

    # Compute the offsets for the current block
    offs_m = pid_m
    offs_n = pid_n + tl.arange(0, TILE_N)
    X_block_ptr = X + (offs_m * N + offs_n)
    Y_block_ptr = Y + (offs_m * N + offs_n)

    # Load the data into the Triton block
    x = tl.load(X_block_ptr, mask=offs_n < N, other=-float('inf'))

    # Compute the maximum value in the block
    max_val = tl.max(x, axis=0)

    # Subtract the maximum value for numerical stability
    x = x - max_val

    # Compute the exponentials
    z = tl.exp(x)

    # Compute the sum of the exponentials
    z_sum = tl.sum(z, axis=0)

    # Normalize the exponentials
    z = z / z_sum

    # Store the results back to the output tensor
    tl.store(Y_block_ptr, z, mask=offs_n < N)

def prev_multiple_of(a, b):
    return (a // b) * b

import torch

def softmax(x: torch.Tensor):
    M, N = x.shape
    out = torch.empty_like(x)
    TILE_N = 128  # Example tile size, can be adjusted

    # Ensure the tile size is a power of 2
    assert (TILE_N & (TILE_N - 1)) == 0, "TILE_N must be a power of 2"

    # Launch the Triton kernel
    grid = (M * ((N + TILE_N - 1) // TILE_N),)
    softmax_kernel_online_v2[grid](x, out, M, N, TILE_N)

    return out
