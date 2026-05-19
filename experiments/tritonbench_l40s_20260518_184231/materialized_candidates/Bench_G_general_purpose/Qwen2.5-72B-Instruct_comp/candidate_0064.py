import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,  # Pointer to input matrix
    lora_ptr,   # Pointer to LoRA weight matrix
    out_ptr,    # Pointer to output matrix
    lora_indices,  # Pointer to LoRA indices
    batch_size,  # Batch size
    hidden_size,  # Hidden size
    lora_size,    # LoRA size
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_K: tl.constexpr,  # Block size for K dimension
    SPLIT_N: tl.constexpr,  # Number of splits for N dimension
    EVEN_K: tl.constexpr,   # Whether K is evenly divisible by BLOCK_K
    ADD_INPUTS: tl.constexpr,  # Whether to add inputs to the output
    CAST_TYPE: tl.constexpr  # Type casting behavior
):
    pid = tl.program_id(axis=0)
    num_blocks_n = (hidden_size + BLOCK_N - 1) // BLOCK_N
    block_id_n = pid % num_blocks_n
    block_id_k = pid // num_blocks_n

    start_n = block_id_n * BLOCK_N
    start_k = block_id_k * BLOCK_K

    end_n = min(start_n + BLOCK_N, hidden_size)
    end_k = min(start_k + BLOCK_K, lora_size)

    input_offset = tl.arange(0, BLOCK_K) + start_k
    lora_offset = tl.arange(0, BLOCK_N) + start_n

    input_block = tl.load(input_ptr + input_offset)
    lora_block = tl.load(lora_ptr + lora_offset[:, None] * hidden_size + input_offset[None, :])

    output_block = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i in range(SPLIT_N):
        start_n_i = i * (hidden_size // SPLIT_N)
        end_n_i = (i + 1) * (hidden_size // SPLIT_N)
        lora_block_i = lora_block[start_n_i:end_n_i, :]
        input_block_i = input_block[start_n_i:end_n_i]

        output_block += tl.dot(lora_block_i, input_block_i)

    if ADD_INPUTS:
        output_block += tl.load(out_ptr + lora_offset)

    if CAST_TYPE:
        output_block = output_block.to(tl.float16)

    tl.store(out_ptr + lora_offset, output_block)

import torch
import triton
from triton.runtime import launch

@torch.inference_mode()
def _bgmv_expand_slice(input, lora, lora_indices, batch_size, hidden_size, lora_size, block_n, block_k, split_n, add_inputs, cast_type):
    # Validate input shapes
    assert input.shape == (batch_size, hidden_size), "Input shape mismatch"
    assert lora.shape == (hidden_size, lora_size), "LoRA weight shape mismatch"
    assert lora_indices.shape == (batch_size,), "LoRA indices shape mismatch"

    # Initialize output tensor
    output = torch.zeros((batch_size, hidden_size), device=input.device, dtype=input.dtype)

    # Define grid configuration
    grid = (batch_size * (hidden_size + block_n - 1) // block_n,)

    # Launch Triton kernel
    _bgmv_expand_slice_kernel[grid](
        input_ptr=input,
        lora_ptr=lora,
        out_ptr=output,
        lora_indices=lora_indices,
        batch_size=batch_size,
        hidden_size=hidden_size,
        lora_size=lora_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_N=split_n,
        EVEN_K=(hidden_size % block_k == 0),
        ADD_INPUTS=add_inputs,
        CAST_TYPE=cast_type
    )

    return output
