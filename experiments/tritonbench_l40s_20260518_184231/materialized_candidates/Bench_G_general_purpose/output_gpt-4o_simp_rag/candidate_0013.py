import triton
import triton.language as tl
import torch

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, out_ptr,
    lora_indices_ptr, scaling,
    xm_stride, xk_stride,
    lm_stride, lk_stride,
    om_stride,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_batches = tl.num_programs(axis=0)

    # Compute batch index
    batch_idx = pid // (N // BLOCK_N)
    block_n_idx = pid % (N // BLOCK_N)

    # Pointers to the start of each batch
    input_batch_ptr = input_ptr + batch_idx * xm_stride
    out_batch_ptr = out_ptr + batch_idx * om_stride
    lora_idx = tl.load(lora_indices_ptr + batch_idx)
    lora_batch_ptr = lora_ptr + lora_idx * lm_stride

    # Offsets for the current block
    input_offset_n = block_n_idx * BLOCK_N
    lora_offset_n = block_n_idx * BLOCK_N

    # Accumulator for the result
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k_block in range(0, K, BLOCK_K):
        input_block_ptr = input_batch_ptr + input_offset_n + k_block * xk_stride
        lora_block_ptr = lora_batch_ptr + lora_offset_n + k_block * lk_stride

        # Load input and lora blocks
        input_block = tl.load(input_block_ptr + tl.arange(0, BLOCK_N))
        lora_block = tl.load(lora_block_ptr + tl.arange(0, BLOCK_N))

        # Compute the product and accumulate
        acc += input_block * lora_block

    # Apply scaling
    acc *= scaling

    # Store the result
    out_block_ptr = out_batch_ptr + input_offset_n
    tl.store(out_block_ptr + tl.arange(0, BLOCK_N), acc)

def _bgmv_shrink(input, lora, lora_indices, scaling, BLOCK_N, BLOCK_K, SPLIT_K):
    # Ensure inputs are contiguous
    input = input.contiguous()
    lora = lora.contiguous()
    lora_indices = lora_indices.contiguous()

    # Determine the batch size
    batch_size = input.shape[0]

    # Calculate grid dimensions
    grid = (batch_size * (lora.shape[1] // BLOCK_N),)

    # Launch the kernel
    _bgmv_shrink_kernel[grid](
        input_ptr=input,
        lora_ptr=lora,
        out_ptr=torch.empty_like(input),
        lora_indices_ptr=lora_indices,
        scaling=scaling,
        xm_stride=input.stride(0),
        xk_stride=input.stride(1),
        lm_stride=lora.stride(0),
        lk_stride=lora.stride(1),
        om_stride=input.stride(0),
        N=lora.shape[1],
        K=lora.shape[0],
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K
    )
