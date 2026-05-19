import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,  # Pointer to the input data
    lora_ptr,   # Pointer to the lora weight matrix
    out_ptr,    # Pointer to the output tensor
    N,          # Number of columns in the lora matrix
    K,          # Number of rows in the lora matrix
    lora_indices,  # Indices indicating which lora matrix to use for each batch
    scaling,    # Scaling factor applied to the result
    xm_stride,  # Stride for input tensor
    xk_stride,  # Stride for input tensor
    lora_n_stride,  # Stride for lora matrix
    lora_k_stride,  # Stride for lora matrix
    out_m_stride,   # Stride for output tensor
    out_n_stride,   # Stride for output tensor
    BLOCK_N: tl.constexpr,  # Tile size for columns
    BLOCK_K: tl.constexpr,  # Tile size for rows
    SPLIT_K: tl.constexpr,  # Number of splits for reduction
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (N // BLOCK_N)
    block_n_id = pid % (N // BLOCK_N)

    lora_index = lora_indices[batch_id]
    lora_ptr = lora_ptr + lora_index * N * K

    for split_k_id in range(SPLIT_K):
        block_k_id = split_k_id * (K // BLOCK_K)
        input_block_ptr = input_ptr + batch_id * K + block_k_id * BLOCK_K
        lora_block_ptr = lora_ptr + block_k_id * BLOCK_K * lora_n_stride

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            input_block = tl.load(input_block_ptr + k * xk_stride, mask=k + tl.arange(0, BLOCK_K) < K)
            lora_block = tl.load(lora_block_ptr + k * lora_k_stride, mask=k + tl.arange(0, BLOCK_K) < K)
            acc += tl.dot(input_block, lora_block)

        out_block_ptr = out_ptr + batch_id * N + block_n_id * BLOCK_N * out_n_stride
        if SPLIT_K > 1:
            tl.atomic_add(out_block_ptr, acc * scaling)
        else:
            tl.store(out_block_ptr, acc * scaling)

import torch
import triton
import triton.language as tl

def _bgmv_shrink(input, lora, out, lora_indices, scaling, BLOCK_N=128, BLOCK_K=64, SPLIT_K=1):
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert lora.is_contiguous(), "Lora tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"

    B, K = input.shape
    N = lora.shape[1]

    # Determine grid dimensions
    grid = (B * (N // BLOCK_N),)

    # Launch the kernel
    _bgmv_shrink_kernel[grid](
        input_ptr=input,
        lora_ptr=lora,
        out_ptr=out,
        N=N,
        K=K,
        lora_indices=lora_indices,
        scaling=scaling,
        xm_stride=input.stride(0),
        xk_stride=input.stride(1),
        lora_n_stride=lora.stride(1),
        lora_k_stride=lora.stride(2),
        out_m_stride=out.stride(0),
        out_n_stride=out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K
    )
