import triton
import triton.language as tl
import torch
import math

@triton.jit
def _bgmv_shrink_kernel(
    # Pointers to matrices
    input_ptr, lora_ptr, out_ptr,
    # Matrix dimensions
    N: tl.constexpr, K: tl.constexpr,
    # Array strides
    input_batch_stride, input_row_stride,
    lora_batch_stride, lora_row_stride,
    # Metadata
    lora_indices_ptr,
    scaling,
    # Grid-related variables
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)  # Batch index
    kid = tl.program_id(1)  # Block index in K dimension

    # Load lora index for this batch
    lora_idx = tl.load(lora_indices_ptr + pid)
    
    # Skip if lora_idx is -1
    if lora_idx == -1:
        return

    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Compute memory offsets
    input_offset = pid * input_batch_stride
    lora_offset = lora_idx * lora_batch_stride
    
    # Iterate over K dimension
    k_start = kid * BLOCK_K
    k_end = min(K, (kid + 1) * BLOCK_K)
    
    for k in range(k_start, k_end):
        # Load input vector element
        x = tl.load(input_ptr + input_offset + k * input_row_stride)
        
        # Load LORA matrix row
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        lora_row = tl.load(lora_ptr + lora_offset + k * lora_row_stride + offs_n, mask=mask_n, other=0.0)
        
        # Compute partial product and accumulate
        acc = acc + x * lora_row
    
    # Apply scaling factor
    acc = acc * scaling
    
    # Store results with atomic add if SPLIT_K > 1
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    if SPLIT_K > 1:
        tl.atomic_add(out_ptr + pid * N + offs_n, acc, mask=mask_n)
    else:
        tl.store(out_ptr + pid * N + offs_n, acc, mask=mask_n)

def _bgmv_shrink(input_tensor, lora_weight, lora_indices, scaling=1.0):
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    lora_weight = lora_weight.contiguous()
    
    # Extract dimensions
    batch_size = input_tensor.shape[0]
    K = input_tensor.shape[1]
    N = lora_weight.shape[1]
    
    # Compute optimal block sizes
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = 32  # Can be tuned based on hardware
    
    # Initialize output tensor
    output = torch.zeros((batch_size, N), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Compute number of K splits
    SPLIT_K = math.ceil(K / BLOCK_K)
    
    # Configure grid
    grid = (batch_size, SPLIT_K)
    
    # Launch kernel
    _bgmv_shrink_kernel[grid](
        input_tensor, lora_weight, output,
        N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        lora_weight.stride(0), lora_weight.stride(1),
        lora_indices,
        scaling,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        BLOCK_N=BLOCK_N,
    )
    
    return output
