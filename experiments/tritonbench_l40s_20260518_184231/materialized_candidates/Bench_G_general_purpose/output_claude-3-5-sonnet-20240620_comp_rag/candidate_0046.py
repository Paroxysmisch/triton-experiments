import torch
import triton
import triton.language as tl
import math

@triton.jit
def max_kernel_1(
    inp_ptr,  # pointer to input tensor
    mid_ptr,  # pointer to intermediate results
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,  # size of each block
):
    # Get program ID and compute offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load values with masking
    x = tl.load(inp_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for this block
    max_val = tl.max(x)
    
    # Store result
    tl.store(mid_ptr + pid, max_val)

@triton.jit
def max_kernel_2(
    mid_ptr,  # pointer to intermediate results
    out_ptr,  # pointer to output tensor
    n_blocks,  # number of blocks from first kernel
    BLOCK_SIZE: tl.constexpr,
):
    # Load all intermediate results
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks
    x = tl.load(mid_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute final max
    max_val = tl.max(x)
    
    # Store final result
    tl.store(out_ptr, max_val)

@triton.jit
def max_kernel(
    inp_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    out_idx_ptr,  # pointer to output indices
    M,  # size of dimension to reduce
    K,  # product of other dimensions
    stride_m,  # stride for dimension to reduce
    stride_k,  # stride for other dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Compute offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    # Initialize max values and indices
    max_vals = tl.full([BLOCK_SIZE_K], -float('inf'), dtype=tl.float32)
    max_idxs = tl.zeros([BLOCK_SIZE_K], dtype=tl.int32)
    
    # Load and compute max values
    for m in range(0, BLOCK_SIZE_M):
        if m < M:
            offs = offs_k * stride_k + m * stride_m
            x = tl.load(inp_ptr + offs, mask=mask_k, other=-float('inf'))
            
            # Update max values and indices
            max_mask = x > max_vals
            max_vals = tl.where(max_mask, x, max_vals)
            max_idxs = tl.where(max_mask, m, max_idxs)
    
    # Store results
    tl.store(out_ptr + offs_k, max_vals, mask=mask_k)
    tl.store(out_idx_ptr + offs_k, max_idxs, mask=mask_k)

def max(x: torch.Tensor) -> torch.Tensor:
    """Compute maximum value of entire tensor."""
    # Determine block size and number of blocks
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(4096, n_elements))
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate intermediate and output tensors
    mid = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    out = torch.empty((), dtype=x.dtype, device=x.device)
    
    # Launch kernels
    max_kernel_1[(n_blocks,)](
        x.data_ptr(),
        mid.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    max_kernel_2[(1,)](
        mid.data_ptr(),
        out.data_ptr(),
        n_blocks,
        BLOCK_SIZE=triton.next_power_of_2(n_blocks),
    )
    
    return out

def max_dim(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute maximum value along specified dimension."""
    # Validate dimension
    dim = dim if dim >= 0 else x.dim() + dim
    assert 0 <= dim < x.dim(), f"Invalid dimension {dim} for tensor of rank {x.dim()}"
    
    # Calculate sizes
    M = x.size(dim)
    K = x.numel() // M
    
    # Calculate strides
    stride_m = x.stride(dim)
    stride_k = x.stride(0) if dim != 0 else x.stride(1)
    
    # Allocate output tensors
    out_shape = list(x.shape)
    out_shape[dim] = 1
    values = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    
    # Calculate grid and block sizes
    BLOCK_SIZE_M = triton.next_power_of_2(min(M, 1024))
    BLOCK_SIZE_K = triton.next_power_of_2(min(K, 1024))
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
    
    # Launch kernel
    max_kernel[grid](
        x.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        M,
        K,
        stride_m,
        stride_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return values.squeeze(dim), indices.squeeze(dim)
