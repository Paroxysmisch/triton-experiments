triton
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr, lora_ptr, out_ptr,
    N, K, lora_indices,
    stride_input_n, stride_input_k,
    stride_lora_n, stride_lora_k,
    stride_out_n, stride_out_k,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    # Determine the batch index and indices within the block
    batch_idx = tl.program_id(0)
    n_idx = batch_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    k_idx = tl.arange(0, BLOCK_K)

    # Determine the input, lora, and output pointers for the current block
    input_block_ptr = input_ptr + n_idx[:, None] * stride_input_n + k_idx[None, :] * stride_input_k
    lora_block_ptr = lora_ptr + n_idx[:, None] * stride_lora_n + k_idx[None, :] * stride_lora_k
    out_block_ptr = out_ptr + n_idx[:, None] * stride_out_n + k_idx[None, :] * stride_out_k

    # Load the input, lora, and lora_indices data
    input_block = tl.load(input_block_ptr, mask=n_idx[:, None] < N, other=0.0)
    lora_block = tl.load(lora_block_ptr, mask=n_idx[:, None] < N, other=0.0)
    lora_indices_block = tl.load(lora_indices + n_idx[:, None] * stride_lora_n, mask=n_idx[:, None] < N, other=-1)

    # Initialize the output block
    output_block = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    # Perform the GEMV for each batch element
    for i in range(BLOCK_N):
        if lora_indices_block[i] == -1:
            continue
        lora_idx = int(lora_indices_block[i])
        lora_row = tl.load(lora_ptr + lora_idx * stride_lora_n + k_idx[None, :] * stride_lora_k, mask=k_idx < K, other=0.0)
        output_block[i, :] += input_block[i, :] @ lora_row

    # Cast the output block if needed
    if CAST_TYPE:
        output_block = tl.bitcast(output_block, tl.float16)

    # Store the output block back to global memory
    tl.store(out_block_ptr, output_block, mask=n_idx[:, None] < N)
