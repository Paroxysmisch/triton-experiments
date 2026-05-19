import triton
import triton.language as tl
import torch

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,           # float*      [B, K]
    lora_ptr,            # float*      [num_lora, N, K]
    out_ptr,             # float*      [B, N]
    lora_indices_ptr,    # int32*      [B]
    # strides
    input_stride_b,      # int
    input_stride_k,      # int
    lora_stride_l,       # int
    lora_stride_n,       # int
    lora_stride_k,       # int
    out_stride_b,        # int
    out_stride_n,        # int
    # sizes
    B,                   # int
    num_lora,            # int
    N,                   # int
    K,                   # int
    scaling,             # float
    # compile-time constants
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_split_k = tl.program_id(2)

    # If batch index is out of range, exit
    if pid_b >= B:
        return

    # Read the LORA index for this batch
    lora_idx = tl.load(lora_indices_ptr + pid_b)
    # Skip if index == -1
    if lora_idx < 0:
        return

    # Each program processes a block of the N dimension
    n_start = pid_n * BLOCK_N
    # Each program processes a portion of K dimension depending on SPLIT_K
    k_chunk_size = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_split_k * k_chunk_size
    k_end = tl.minimum(k_start + k_chunk_size, K)

    # Create an accumulator for the block of N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in small chunks of BLOCK_K for better data loading
    # We'll accumulate partial sums in 'acc'
    for kk in range(k_start, k_end, BLOCK_K):
        # Current block size we will process (may be partial at the end)
        curr_k_size = tl.minimum(K - kk, BLOCK_K)

        # For each k in [kk, kk + curr_k_size)
        #   load input[b, k]
        #   load lora[lora_idx, n, k]
        #   multiply and accumulate for each n
        # We'll load input and LORA one element at a time in this example
        # (a more performant version would vectorize loads/stores)
        for k_offset in range(curr_k_size):
            k_idx = kk + k_offset
            in_val = tl.load(input_ptr + pid_b * input_stride_b + k_idx * input_stride_k)

            # Accumulate for each n in the block
            for i in range(BLOCK_N):
                n_idx = n_start + i
                if n_idx < N:
                    lora_val = tl.load(
                        lora_ptr
                        + lora_idx * lora_stride_l
                        + n_idx * lora_stride_n
                        + k_idx * lora_stride_k
                    )
                    acc[i] += in_val * lora_val

    # Write result back to out tensor. If SPLIT_K > 1, use atomic add
    for i in range(BLOCK_N):
        n_idx = n_start + i
        if n_idx < N:
            dst_ptr = out_ptr + pid_b * out_stride_b + n_idx * out_stride_n
            val = acc[i] * scaling
            # If multiple splits, use atomic add
            if SPLIT_K > 1:
                tl.atomic_add(dst_ptr, val)
            else:
                tl.store(dst_ptr, val)

def _bgmv_shrink(
    input_tensor: torch.Tensor,
    lora_tensor: torch.Tensor,
    lora_indices: torch.Tensor,
    out: torch.Tensor,
    scaling: float = 1.0,
    split_k: int = 1,
):
    """
    Launches the _bgmv_shrink_kernel to perform a batched generalized matrix-vector
    multiplication with low-rank adaptation.
    """

    # Ensure tensors are contiguous
    input_ctg = input_tensor.contiguous()
    lora_ctg = lora_tensor.contiguous()
    out_ctg = out.contiguous()
    lora_indices_ctg = lora_indices.contiguous()

    B = input_ctg.shape[0]
    K = input_ctg.shape[1]
    # lora_ctg shape is [num_lora, N, K]
    num_lora, N, K_lora = lora_ctg.shape
    assert K == K_lora, "Dimension mismatch between input K and LORA K."

    # Strides
    input_stride_b = input_ctg.stride(0)
    input_stride_k = input_ctg.stride(1)

    lora_stride_l = lora_ctg.stride(0)
    lora_stride_n = lora_ctg.stride(1)
    lora_stride_k = lora_ctg.stride(2)

    out_stride_b = out_ctg.stride(0)
    out_stride_n = out_ctg.stride(1)

    # Compute block size for N as a power of
