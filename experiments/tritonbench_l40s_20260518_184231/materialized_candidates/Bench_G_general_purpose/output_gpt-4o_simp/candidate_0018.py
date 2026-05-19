import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(input_ptr, lora_ptr, out_ptr, lora_indices, scaling,
                        N, K, xm_stride, xk_stride, lm_stride, lk_stride, om_stride,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)

    # Determine start of the batch
    lora_idx = lora_indices[batch_id]
    lora_offset = lora_idx * lm_stride

    # Determine the starting positions
    n_start = pid * BLOCK_N
    k_start = 0

    # Create accumulators for the results
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k_offset in range(0, K, BLOCK_K):
        # Load input and lora blocks
        input_block = tl.load(input_ptr + batch_id * xm_stride + k_offset * xk_stride + tl.arange(0, BLOCK_K))
        lora_block = tl.load(lora_ptr + lora_offset + k_offset * lk_stride + tl.arange(0, BLOCK_K))

        # Perform the matrix-vector multiplication
        acc += tl.dot(input_block, lora_block)

    # Apply scaling
    acc *= scaling

    # Store results
    out_offset = batch_id * om_stride + n_start
    if SPLIT_K > 1:
        # Use atomic add for reductions across multiple kernel instances
        tl.atomic_add(out_ptr + out_offset, acc)
    else:
        tl.store(out_ptr + out_offset, acc)

import torch

def _bgmv_shrink(input_tensor, lora_tensor, output_tensor, lora_indices, scaling,
                 BLOCK_N=128, BLOCK_K=128, SPLIT_K=1):
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    lora_tensor = lora_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Determine dimensions
    batch_size = input_tensor.shape[0]
    N = lora_tensor.shape[1]
    K = lora_tensor.shape[0]

    # Strides
    xm_stride = input_tensor.stride(0)
    xk_stride = input_tensor.stride(1)
    lm_stride = lora_tensor.stride(0)
    lk_stride = lora_tensor.stride(1)
    om_stride = output_tensor.stride(0)

    # Grid dimensions
    grid = (triton.cdiv(N, BLOCK_N), batch_size)

    # Launch the kernel
    _bgmv_shrink_kernel[grid](
        input_tensor, lora_tensor, output_tensor, lora_indices, scaling,
        N, K, xm_stride, xk_stride, lm_stride, lk_stride, om_stride,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K
    )
