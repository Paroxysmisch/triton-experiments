import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, out_ptr, lora_indices_ptr,
    input_stride, lora_stride, out_stride,
    B, N, K, BLOCK_K, SPLIT_K, scaling
):
    # Define block indices
    batch_idx = tl.program_id(0)
    block_n = tl.program_id(1)
    block_k = tl.program_id(2)

    # Check if the batch should be processed
    lora_idx = tl.load(lora_indices_ptr + batch_idx)
    if lora_idx == -1:
        return

    # Initialize accumulators
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load input and LORA blocks
        input_offset = batch_idx * input_stride + block_n * N + k
        lora_offset = lora_idx * lora_stride + k * BLOCK_K
        input_block = tl.load(input_ptr + input_offset)
        lora_block = tl.load(lora_ptr + lora_offset)

        # Perform element-wise multiplication and reduction
        acc += input_block * lora_block

    # Apply scaling
    acc *= scaling

    # Write results back to output
    out_offset = batch_idx * out_stride + block_n * N
    if SPLIT_K > 1:
        tl.atomic_add(out_ptr + out_offset, acc)
    else:
        tl.store(out_ptr + out_offset, acc)

import torch

def _bgmv_shrink(input, lora, out, lora_indices, scaling, BLOCK_K=32, SPLIT_K=1):
    assert input.is_contiguous()
    assert lora.is_contiguous()
    assert out.is_contiguous()

    B, N, K = lora.shape
    BLOCK_N = 2 ** (N - 1).bit_length()  # Next power of 2 greater than or equal to N

    grid = (B, (N + BLOCK_N - 1) // BLOCK_N, SPLIT_K)

    input_ptr = input.data_ptr()
    lora_ptr = lora.data_ptr()
    out_ptr = out.data_ptr()
    lora_indices_ptr = lora_indices.data_ptr()

    input_stride = input.stride(0)
    lora_stride = lora.stride(0)
    out_stride = out.stride(0)

    _bgmv_shrink_kernel[grid](
        input_ptr, lora_ptr, out_ptr, lora_indices_ptr,
        input_stride, lora_stride, out_stride,
        B, N, K, BLOCK_K, SPLIT_K, scaling
    )
