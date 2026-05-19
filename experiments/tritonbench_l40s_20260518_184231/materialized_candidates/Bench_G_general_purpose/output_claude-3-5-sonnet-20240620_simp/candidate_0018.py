import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, out_ptr,
    N, K,
    lora_indices,
    scaling,
    xm_stride, xk_stride,
    wm_stride, wk_stride,
    om_stride,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    batch_id = pid // SPLIT_K
    split_k_id = pid % SPLIT_K

    # Compute the starting offset for this batch
    lora_idx = tl.load(lora_indices + batch_id)
    
    # Initialize pointers
    input_block_ptr = input_ptr + batch_id * xm_stride
    lora_block_ptr = lora_ptr + lora_idx * wm_stride
    output_block_ptr = out_ptr + batch_id * om_stride

    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over K dimension in steps of BLOCK_K
    for k in range(split_k_id, K, SPLIT_K):
        k_remaining = min(BLOCK_K, K - k)
        
        # Load input block
        x = tl.load(input_block_ptr + k * xk_stride + tl.arange(0, k_remaining))
        
        # Load lora weight block
        w = tl.load(lora_block_ptr + k * wk_stride + tl.arange(0, k_remaining))
        
        # Compute partial product
        acc += tl.sum(x * w, axis=0)

    # Apply scaling factor
    acc = acc * scaling

    # Store result with atomic add if using SPLIT_K > 1
    if SPLIT_K > 1:
        tl.atomic_add(output_block_ptr + tl.arange(0, BLOCK_N), acc)
    else:
        tl.store(output_block_ptr + tl.arange(0, BLOCK_N), acc)

def _bgmv_shrink(x, w, lora_indices, scaling):
    """
    Wrapper function for the BGMV shrink kernel.
    
    Args:
        x: Input tensor
        w: Lora weights tensor
        lora_indices: Indices for lora matrix selection
        scaling: Scaling factor for the output
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    lora_indices = lora_indices.contiguous()
    
    # Get dimensions
    batch_size = x.shape[0]
    K = w.shape[1]
    N = w.shape[0]
    
    # Create output tensor
    output = torch.zeros((batch_size, N), device=x.device, dtype=x.dtype)
    
    # Define block sizes
    BLOCK_N = 32
    BLOCK_K = 32
    SPLIT_K = 4
    
    # Calculate grid size
    grid = (batch_size * SPLIT_K,)
    
    # Launch kernel
    _bgmv_shrink_kernel[grid](
        x, w, output,
        N, K,
        lora_indices,
        scaling,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        output.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K
    )
    
    return output
