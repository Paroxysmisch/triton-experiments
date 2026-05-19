import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, input_batch_stride, input_n_stride, input_k_stride,
    lora_ptr, lora_batch_stride, lora_n_stride, lora_k_stride,
    lora_indices_ptr, lora_indices_stride,
    out_ptr, out_batch_stride, out_n_stride,
    N, K, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, scaling: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_batches = input_batch_stride // input_n_stride
    batch_id = pid // N
    n_id = pid % N

    if batch_id >= num_batches:
        return

    lora_index = tl.load(lora_indices_ptr + batch_id * lora_indices_stride)
    if lora_index == -1:
        return

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for split_k_id in range(SPLIT_K):
        k_start = split_k_id * BLOCK_K
        k_end = min(k_start + BLOCK_K, K)

        input_block_ptr = input_ptr + batch_id * input_batch_stride + n_id * input_n_stride + k_start * input_k_stride
        lora_block_ptr = lora_ptr + lora_index * lora_batch_stride + n_id * lora_n_stride + k_start * lora_k_stride

        for k in range(k_start, k_end):
            input_val = tl.load(input_block_ptr + k * input_k_stride)
            lora_val = tl.load(lora_block_ptr + k * lora_k_stride)
            acc += input_val * lora_val

    if SPLIT_K > 1:
        out_block_ptr = out_ptr + batch_id * out_batch_stride + n_id * out_n_stride
        tl.atomic_add(out_block_ptr, acc * scaling)
    else:
        out_block_ptr = out_ptr + batch_id * out_batch_stride + n_id * out_n_stride
        tl.store(out_block_ptr, acc * scaling)

import torch

def _bgmv_shrink(
    input: torch.Tensor,
    lora: torch.Tensor,
    lora_indices: torch.Tensor,
    scaling: float,
    out: Optional[torch.Tensor] = None,
    BLOCK_K: int = 128,
    SPLIT_K: int = 1
):
    # Ensure input, lora, and out tensors are contiguous
    input = input.contiguous()
    lora = lora.contiguous()
    if out is None:
        out = torch.zeros_like(input)
    else:
        out = out.contiguous()

    # Extract dimensions
    num_batches = input.size(0)
    N = input.size(1)
    K = input.size(2)

    # Compute BLOCK_N
    BLOCK_N = 1
    while BLOCK_N < N:
        BLOCK_N *= 2

    # Configure grid
    grid = (num_batches * N,)

    # Launch kernel
    _bgmv_shrink_kernel[grid](
        input_ptr=input,
        input_batch_stride=input.stride(0),
        input_n_stride=input.stride(1),
        input_k_stride=input.stride(2),
        lora_ptr=lora,
        lora_batch_stride=lora.stride(0),
        lora_n_stride=lora.stride(1),
        lora_k_stride=lora.stride(2),
        lora_indices_ptr=lora_indices,
        lora_indices_stride=lora_indices.stride(0),
        out_ptr=out,
        out_batch_stride=out.stride(0),
        out_n_stride=out.stride(1),
        N=N,
        K=K,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        scaling=scaling
    )

    return out
