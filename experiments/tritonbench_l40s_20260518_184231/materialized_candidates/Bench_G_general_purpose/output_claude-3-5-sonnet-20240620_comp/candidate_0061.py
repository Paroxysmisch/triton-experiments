import torch
import triton
import triton.language as tl

# Define block sizes for the kernel
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

@triton.jit
def _sgmv_expand_slice_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, lora_indices_ptr, output_ptr,
    # Matrix dimensions
    B, M, N, K,
    # Strides for the tensors
    stride_xb, stride_xm, stride_xk,
    stride_wb, stride_wm, stride_wn,
    stride_out_b, stride_out_m,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    # Calculate the offsets
    # Block offset for batch dimension
    batch_idx = bid
    # Block offset for M dimension
    m_offset = pid * BLOCK_M

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Get the lora index for this batch
    lora_idx = tl.load(lora_indices_ptr + batch_idx)

    # Iterate over blocks in the K dimension
    for k in range(0, K, BLOCK_K):
        # Load x block
        x_block_ptr = x_ptr + batch_idx * stride_xb + m_offset * stride_xm + k
        x_mask = tl.arange(0, BLOCK_K) < (K - k)
        x = tl.load(x_block_ptr + tl.arange(0, BLOCK_K) * stride_xk, mask=x_mask, other=0.0)

        # Load weight block using lora_idx
        w_block_ptr = weight_ptr + lora_idx * stride_wb + m_offset * stride_wm + k
        w = tl.load(w_block_ptr + tl.arange(0, BLOCK_K) * stride_wn, mask=x_mask, other=0.0)

        # Compute matrix multiplication
        acc += tl.sum(x * w, axis=0)

    # Write output
    out_ptr = output_ptr + batch_idx * stride_out_b + m_offset
    mask = tl.arange(0, BLOCK_M) < M
    tl.store(out_ptr + tl.arange(0, BLOCK_M) * stride_out_m, acc, mask=mask)

def _sgmv_expand_slice(x, weight, lora_indices):
    """
    Wrapper function for the SGMV kernel with LoRA weight slicing.
    
    Args:
        x: Input tensor of shape (B, M, K)
        weight: Weight tensor of shape (L, M, K) where L is number of LoRA layers
        lora_indices: Tensor of shape (B,) containing indices into the weight tensor
    
    Returns:
        output: Tensor of shape (B, M)
    """
    # Get dimensions
    B, M, K = x.shape
    L, M_w, K_w = weight.shape
    
    # Validate shapes
    assert M == M_w and K == K_w, "Input and weight dimensions must match"
    assert lora_indices.shape == (B,), "LoRA indices must match batch size"
    
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    lora_indices = lora_indices.contiguous()
    
    # Compute output
    output = torch.empty((B, M), device=x.device, dtype=x.dtype)
    
    # Configure grid
    grid = lambda meta: (
        triton.cdiv(M, BLOCK_M),  # Blocks in M dimension
        B,                         # Blocks for batch dimension
    )
    
    # Launch kernel
    _sgmv_expand_slice_kernel[grid](
        x_ptr=x, 
        weight_ptr=weight,
        lora_indices_ptr=lora_indices,
        output_ptr=output,
        B=B, M=M, N=N, K=K,
        stride_xb=x.stride(0),
        stride_xm=x.stride(1),
        stride_xk=x.stride(2),
        stride_wb=weight.stride(0),
        stride_wm=weight.stride(1),
        stride_wn=weight.stride(2),
        stride_out_b=output.stride(0),
        stride_out_m=output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output
