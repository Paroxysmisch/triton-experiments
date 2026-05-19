import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr, lora_ptr, out_ptr, lora_indices,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr, ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr
):
    # Pointers to the start of each block
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Block start indices
    block_n_start = pid_n * BLOCK_N
    block_k_start = pid_k * BLOCK_K

    # Load indices for the current batch
    lora_index = tl.load(lora_indices + pid_n)

    # Initialize accumulation
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, BLOCK_K, SPLIT_N):
        # Load inputs
        input_offset = block_k_start + k
        input_vals = tl.load(input_ptr + input_offset)

        # Load LoRA weights
        lora_offset = lora_index * BLOCK_K + k
        lora_vals = tl.load(lora_ptr + lora_offset)

        # Compute matrix-vector product for the block
        acc += input_vals * lora_vals

    # Optionally add input values to the output
    if ADD_INPUTS:
        acc += tl.load(out_ptr + block_n_start)

    # Cast the result if needed
    if CAST_TYPE:
        acc = acc.to(CAST_TYPE)

    # Store the result in the output pointer
    tl.store(out_ptr + block_n_start, acc)

import torch

@torch.inference_mode()
def _bgmv_expand_slice(input_tensor, lora_tensor, lora_indices, BLOCK_N, BLOCK_K, SPLIT_N, EVEN_K, ADD_INPUTS, CAST_TYPE):
    # Validate shapes and types
    assert input_tensor.shape[1] % BLOCK_K == 0, "Input K dimension must be divisible by BLOCK_K"
    assert lora_tensor.shape[0] == lora_indices.shape[0], "LoRA indices must match LoRA tensor batch size"

    # Get tensor properties
    batch_size, hidden_size = input_tensor.shape

    # Prepare output tensor
    out_tensor = torch.zeros_like(input_tensor)

    # Grid configuration
    grid = (hidden_size // BLOCK_N, batch_size // BLOCK_K)

    # Launch the Triton kernel
    _bgmv_expand_slice_kernel[grid](
        input_tensor, lora_tensor, out_tensor, lora_indices,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_N=SPLIT_N,
        EVEN_K=EVEN_K, ADD_INPUTS=ADD_INPUTS, CAST_TYPE=CAST_TYPE
    )

    return out_tensor
