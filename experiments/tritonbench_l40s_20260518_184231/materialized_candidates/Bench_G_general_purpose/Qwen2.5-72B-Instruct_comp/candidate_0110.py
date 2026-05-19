import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr, lora_ptr, out_ptr, lora_indices,
    N, K, B, stride_input_b, stride_input_k, stride_lora_b, stride_lora_k, stride_lora_n, stride_out_b, stride_out_k,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, CAST_TYPE: tl.constexpr
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    # Compute the start and end indices for the block
    start_n = pid * BLOCK_N
    end_n = start_n + BLOCK_N

    # Load the LoRA index for the batch
    lora_index = tl.load(lora_indices + bid)

    # Skip computation if the LoRA index is -1
    if lora_index == -1:
        return

    # Compute the base pointers for the input and LoRA weights
    input_base = input_ptr + bid * stride_input_b
    lora_base = lora_ptr + bid * stride_lora_b + lora_index * stride_lora_n

    # Iterate over the blocks of N
    for n in range(start_n, end_n):
        if n >= N:
            break

        # Initialize the output value for this block
        out_val = tl.zeros((BLOCK_K,), dtype=tl.float32)

        # Iterate over the blocks of K
        for k in range(0, K, BLOCK_K):
            # Load the input and LoRA weights
            input_vals = tl.load(input_base + k, mask=k + tl.arange(0, BLOCK_K) < K, other=0.0)
            lora_vals = tl.load(lora_base + k * stride_lora_k + n * stride_lora_n, mask=k + tl.arange(0, BLOCK_K) < K, other=0.0)

            # Perform the matrix-vector multiplication
            out_val += input_vals * lora_vals

        # Cast the output value if needed
        if CAST_TYPE:
            out_val = out_val.to(tl.float16)

        # Store the result in the output matrix
        out_base = out_ptr + bid * stride_out_b + n * stride_out_k
        tl.store(out_base, out_val, mask=tl.arange(0, BLOCK_K) < K)

import torch
import triton
import triton.language as tl

def _bgmv_expand(input, lora_weights, lora_indices, out, split_n=1):
    # Ensure the input tensors are contiguous
    input = input.contiguous()
    lora_weights = lora_weights.contiguous()
    lora_indices = lora_indices.contiguous()
    out = out.contiguous()

    # Get the dimensions
    B, N, K = input.shape[0], input.shape[1], input.shape[2]
    lora_N = lora_weights.shape[2]

    # Determine the block sizes
    BLOCK_K = 16
    BLOCK_N = 32

    # Determine if type casting is needed
    CAST_TYPE = input.dtype != lora_weights.dtype

    # Set up the grid configuration
    grid = (N // BLOCK_N, B)

    # Launch the Triton kernel
    _bgmv_expand_kernel[grid](
        input, lora_weights, out, lora_indices,
        N, K, B, input.stride(0), input.stride(2), lora_weights.stride(0), lora_weights.stride(1), lora_weights.stride(2), out.stride(0), out.stride(2),
        BLOCK_N, BLOCK_K, CAST_TYPE
    )

    return out
