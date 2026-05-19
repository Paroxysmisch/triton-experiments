import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,  # Pointer to the input tensor
    lora_ptr,   # Pointer to the LORA matrix
    out_ptr,    # Pointer to the output tensor
    lora_indices,  # Indices for LORA matrices
    N, K,       # Dimensions of the LORA matrix
    scaling,    # Scaling factor
    stride_input_b, stride_input_k,  # Strides for input tensor
    stride_lora_b, stride_lora_n, stride_lora_k,  # Strides for LORA matrix
    stride_out_b, stride_out_n,  # Strides for output tensor
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_K: tl.constexpr,  # Block size for K dimension
    SPLIT_K: tl.constexpr,  # Number of splits along K dimension
):
    # Get the batch and block indices
    pid = tl.program_id(0)
    batch = pid // SPLIT_K
    split_k = pid % SPLIT_K

    # Load the LORA index for the current batch
    lora_idx = tl.load(lora_indices + batch)

    # Skip if the LORA index is -1
    if lora_idx == -1:
        return

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Compute the range of K values for this split
    k_start = split_k * BLOCK_K
    k_end = tl.minimum(k_start + BLOCK_K, K)

    # Iterate over the K blocks
    for k in range(k_start, k_end, BLOCK_K):
        # Load the input block
        input_block = tl.load(input_ptr + batch * stride_input_b + k * stride_input_k + tl.arange(0, BLOCK_K))

        # Load the LORA block
        lora_block = tl.load(lora_ptr + lora_idx * stride_lora_b + tl.arange(0, BLOCK_N)[:, None] * stride_lora_n + k * stride_lora_k + tl.arange(0, BLOCK_K))

        # Perform the element-wise multiplication and reduction
        acc += tl.sum(input_block * lora_block, axis=1)

    # Apply the scaling factor
    acc *= scaling

    # Store the result
    if SPLIT_K > 1:
        tl.atomic_add(out_ptr + batch * stride_out_b + tl.arange(0, BLOCK_N) * stride_out_n, acc)
    else:
        tl.store(out_ptr + batch * stride_out_b + tl.arange(0, BLOCK_N) * stride_out_n, acc)

import torch
import triton
import triton.language as tl

def _bgmv_shrink(input, lora, out, lora_indices, scaling):
    # Ensure the tensors are contiguous
    input = input.contiguous()
    lora = lora.contiguous()
    out = out.contiguous()

    # Extract dimensions
    batch_count = lora_indices.size(0)
    N, K = lora.size(1), lora.size(2)

    # Compute BLOCK_N as the next power of two greater than or equal to N
    BLOCK_N = 1
    while BLOCK_N < N:
        BLOCK_N *= 2

    # Define the block size for K dimension
    BLOCK_K = 128  # Example block size, can be tuned

    # Define the number of splits along K dimension
    SPLIT_K = (K + BLOCK_K - 1) // BLOCK_K

    # Configure the grid
    grid = (batch_count * SPLIT_K,)

    # Launch the kernel
    _bgmv_shrink_kernel[grid](
        input, lora, out, lora_indices,
        N, K, scaling,
        input.stride(0), input.stride(1),
        lora.stride(0), lora.stride(1), lora.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_N, BLOCK_K, SPLIT_K
    )
