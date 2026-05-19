import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,  # Pointer to the input tensor
    rms_w_ptr,  # Pointer to the RMS weights
    out_ptr,  # Pointer to the output tensor
    batch,  # Batch size
    M,  # Dimension M
    K,  # Dimension K
    N_SIZE,  # Size of K
    eps,  # Small constant to prevent division by zero
    BLOCK_N_SIZE: tl.constexpr,  # Block size for processing K dimension
    num_warps: tl.constexpr  # Number of warps
):
    pid = tl.program_id(axis=0)  # Get the program ID
    batch_id = pid // M  # Compute the batch ID
    m_id = pid % M  # Compute the M ID

    # Compute the starting index for the current block
    k_start = tl.arange(0, BLOCK_N_SIZE) * (K // BLOCK_N_SIZE)
    k_end = k_start + BLOCK_N_SIZE

    # Initialize the sum of squares
    sum_of_squares = tl.zeros([BLOCK_N_SIZE], dtype=tl.float32)

    # Iterate over the K dimension in chunks of BLOCK_N_SIZE
    for k in range(K // BLOCK_N_SIZE):
        x_block = tl.load(x_ptr + (batch_id * M * K + m_id * K + k_start + k * BLOCK_N_SIZE))
        sum_of_squares += x_block * x_block

    # Compute the RMS value
    rms_value = tl.sqrt(tl.sum(sum_of_squares) / N_SIZE + eps)

    # Load the RMS weights
    rms_weights = tl.load(rms_w_ptr + m_id)

    # Normalize and scale the elements
    out_block = (x_block / rms_value) * rms_weights

    # Store the output
    tl.store(out_ptr + (batch_id * M * K + m_id * K + k_start + k * BLOCK_N_SIZE), out_block)

import torch

def rmsnorm_wrapper(x, rms_weights, eps=1e-6, BLOCK_N_SIZE=128, num_warps=4):
    # Ensure the input tensor is on the GPU
    x = x.cuda()
    rms_weights = rms_weights.cuda()

    # Get the dimensions of the input tensor
    batch, M, K = x.shape
    N_SIZE = K

    # Allocate the output tensor
    out = torch.empty_like(x)

    # Compute the grid dimensions
    grid = (batch * M,)

    # Launch the kernel
    rmsnorm_triton[grid](
        x, rms_weights, out, batch, M, K, N_SIZE, eps, BLOCK_N_SIZE, num_warps
    )

    return out
