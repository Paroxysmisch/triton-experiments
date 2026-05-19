import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, out_ptr,
    N, K, lora_indices, scaling,
    xm_stride, xk_stride, ln_stride, lk_stride, lo_stride, lm_stride,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)

    # Compute the lora index for this batch
    lora_idx = tl.load(lora_indices + batch_id)

    # Compute the base pointers for input, lora, and output
    input_base = input_ptr + batch_id * xm_stride
    lora_base = lora_ptr + lora_idx * ln_stride
    out_base = out_ptr + batch_id * lo_stride

    # Compute the block id in the N dimension
    n_start = pid * BLOCK_N
    n_end = tl.minimum(n_start + BLOCK_N, N)

    # Compute the block id in the K dimension
    k_start = 0
    k_end = K

    # Iterate over the K dimension in blocks
    for k_block in range(0, K, BLOCK_K):
        k_block_end = tl.minimum(k_block + BLOCK_K, K)

        # Load the input block
        input_block = tl.load(input_base + k_block * xk_stride + tl.arange(0, BLOCK_N)[:, None], mask=k_block + tl.arange(0, BLOCK_K) < K, other=0.0)

        # Load the lora block
        lora_block = tl.load(lora_base + k_block * lk_stride + tl.arange(0, BLOCK_N)[:, None], mask=k_block + tl.arange(0, BLOCK_K) < K, other=0.0)

        # Compute the product
        output_block = tl.dot(input_block, lora_block, allow_tf32=True)

        # Apply the scaling factor
        output_block *= scaling

        # Store the result in the output tensor
        if SPLIT_K == 1:
            tl.store(out_base + n_start * lm_stride + tl.arange(0, BLOCK_K)[:, None], output_block, mask=n_start + tl.arange(0, BLOCK_N) < N)
        else:
            tl.atomic_add(out_base + n_start * lm_stride + tl.arange(0, BLOCK_K)[:, None], output_block, mask=n_start + tl.arange(0, BLOCK_N) < N)

        # Move to the next block in the K dimension
        k_start = k_block_end

### Wrapper Function Implementation

import torch

def _bgmv_shrink(
    input: torch.Tensor,
    lora: torch.Tensor,
    out: torch.Tensor,
    lora_indices: torch.Tensor,
    scaling: float,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    SPLIT_K: int = 1
):
    # Ensure inputs are contiguous
    input = input.contiguous()
    lora = lora.contiguous()
    out = out.contiguous()
    lora_indices = lora_indices.contiguous()

    # Determine batch size
    batch_size = input.size(0)
    N = input.size(1)
    K = lora.size(1)

    # Compute grid dimensions
    grid = (triton.cdiv(N, BLOCK_N), batch_size, SPLIT_K)

    # Launch the kernel
    _bgmv_shrink_kernel[grid](
        input, lora, out,
        N, K, lora_indices, scaling,
        input.stride(0), input.stride(1), lora.stride(0), lora.stride(1), out.stride(0), out.stride(1),
        BLOCK_N, BLOCK_K, SPLIT_K
    )
