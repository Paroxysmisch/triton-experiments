import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, out_ptr, lora_indices, scaling,
    N, K, SPLIT_K, BLOCK_K, BLOCK_N, BLOCK_M,
    num_batches, input_stride, lora_stride, out_stride
):
    # Compute indices
    batch_idx = tl.program_id(0)
    split_k_idx = tl.program_id(1)
    n_idx = tl.program_id(2)
    m_idx = tl.program_id(3)

    # Check if the batch should be skipped
    if batch_idx >= num_batches:
        return

    # Get the LORA index for the current batch
    lora_idx = lora_indices[batch_idx]

    # Skip this batch if the LORA index is -1
    if lora_idx == -1:
        return

    # Compute the base addresses for input, LORA, and output
    input_base = input_ptr + batch_idx * input_stride
    lora_base = lora_ptr + batch_idx * lora_stride
    out_base = out_ptr + batch_idx * out_stride

    # Initialize the accumulator
    accumulator = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Iterate over blocks of K
    for k_block in range(SPLIT_K):
        # Compute the current block of K
        k_start = k_block * BLOCK_K
        k_end = min(k_start + BLOCK_K, K)

        # Load the input block
        input_block = tl.load(input_base + n_idx * input_stride + k_start * input_stride, mask=(k_end - k_start) > 0)

        # Load the LORA block
        lora_block = tl.load(lora_base + lora_idx * lora_stride + n_idx * lora_stride + k_start * lora_stride, mask=(k_end - k_start) > 0)

        # Perform element-wise multiplication and reduction
        accumulator += input_block * lora_block

    # Scale the accumulator
    accumulator *= scaling

    # Store the result
    tl.store(out_base + n_idx * out_stride, accumulator, mask=(n_idx < N))
