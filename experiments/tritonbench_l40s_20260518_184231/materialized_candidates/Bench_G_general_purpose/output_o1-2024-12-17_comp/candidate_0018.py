import triton
import triton.language as tl


@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,            # *float32
    lora_ptr,             # *float32
    out_ptr,              # *float32
    lora_indices_ptr,     # *int32
    scaling,              # float32
    batch_size,           # int32
    N,                    # int32
    K,                    # int32
    stride_input_batch,   # int32
    stride_input_k,       # int32
    stride_lora_batch,    # int32
    stride_lora_k,        # int32
    stride_out_batch,     # int32
    SPLIT_K,              # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    b_idx = pid_b

    # Skip if we're out of range
    if b_idx >= batch_size:
        return

    # Check LORA index for this batch
    lora_index = tl.load(lora_indices_ptr + b_idx)
    if lora_index == -1:
        return

    # Create accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_K
    # Each kernel instance will handle SPLIT_K such chunks
    for split_k_id in range(SPLIT_K):
        k_offset = split_k_id * BLOCK_K
        k_range = tl.arange(0, BLOCK_K)
        k_idx = k_offset + k_range

        # Mask for valid k
        k_mask = k_idx < K

        # Load input segment [K]
        inp_ptrs = input_ptr + b_idx * stride_input_batch + k_idx * stride_input_k
        inp = tl.load(inp_ptrs, mask=k_mask, other=0.0)

        # Load LORA segment [N,K]
        # n has shape BLOCK_N, k has shape BLOCK_K
        n_range = tl.arange(0, BLOCK_N)
        n_idx = n_start + n_range
        n_mask = n_idx < N

        # Broadcast pointers
        lora_base = lora_ptr + lora_index * stride_lora_batch
        lora_ptrs = lora_base + n_idx[:, None] * stride_lora_k + k_idx[None, :]
        lora_vals = tl.load(lora_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Multiply and accumulate
        part_sum = tl.sum(lora_vals * inp[None, :], axis=1)
        acc += part_sum

    # Write result
    # If SPLIT_K > 1, we do atomic add; otherwise, direct store
    n_range = tl.arange(0, BLOCK_N)
    n_idx = n_start + n_range
    mask = n_idx < N

    out_ptrs = out_ptr + b_idx * stride_out_batch + n_idx
    if SPLIT_K > 1:
        old = tl.atomic_add(out_ptrs, acc * scaling, mask=mask)
    else:
        tl.store(out_ptrs, acc * scaling, mask=mask)


def _bgmv_shrink(input_ptr, lora_ptr, out_ptr, lora_indices, scaling, batch_size, N, K,
                 stride_input_batch, stride_input_k,
                 stride_lora_batch, stride_lora_k,
                 stride_out_batch, queue=None):
    # Triton uses powers of two or multiples for block size performance
    BLOCK_N = 1
    while BLOCK_N < N:
        BLOCK_N *= 2

    BLOCK_M = 1  # Single row-block for BGMV
    BLOCK_K = 32  # Tune for performance

    # Determine split along K
    SPLIT_K = (K + BLOCK_K - 1) // BLOCK_K

    grid = ( (N + BLOCK_N - 1) // BLOCK_N, batch_size )

    _bgmv_shrink_kernel[grid](
        input_ptr,
        lora_ptr,
        out_ptr,
        lora_indices,
        scaling,
        batch_size,
        N,
        K,
        stride_input_batch,
        stride_input_k,
        stride_lora_batch,
        stride_lora_k,
        stride_out_batch,
        SPLIT_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
        stream=queue
    )
