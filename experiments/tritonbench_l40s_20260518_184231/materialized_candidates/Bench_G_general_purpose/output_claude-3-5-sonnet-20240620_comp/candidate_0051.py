import triton
import triton.language as tl
import torch

@triton.jit
def max_kernel_1(
    x_ptr,          # pointer to input tensor
    mid_ptr,        # pointer to intermediate results
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,  # size of each block
):
    # Program ID gives starting offset for this block
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Create mask for valid memory accesses
    mask = offset + tl.arange(0, BLOCK_SIZE) < n_elements
    
    # Load values with masking
    x = tl.load(x_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=-float('inf'))
    
    # Compute max for this block
    max_val = tl.max(x)
    
    # Store result
    tl.store(mid_ptr + pid, max_val)

@triton.jit
def max_kernel_2(
    mid_ptr,        # pointer to intermediate results
    out_ptr,        # pointer to output tensor
    n_blocks,       # number of blocks to process
):
    # Single thread computes final max
    x = tl.load(mid_ptr + tl.arange(0, n_blocks))
    max_val = tl.max(x)
    tl.store(out_ptr, max_val)

@triton.jit
def max_kernel(
    x_ptr,          # pointer to input tensor
    out_ptr,        # pointer to output values
    idx_ptr,        # pointer to output indices
    stride_dim,     # stride of reduction dimension
    stride_other,   # stride of other dimensions
    size_dim,       # size of reduction dimension
    size_other,     # size of other dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    mask_m = offs_m < size_other
    mask_k = offs_k < size_dim
    
    # Initialize max values and indices
    max_vals = tl.full([BLOCK_SIZE_M], float('-inf'), dtype=tl.float32)
    max_idxs = tl.zeros([BLOCK_SIZE_M], dtype=tl.int32)
    
    # Load and compute max values
    for k in range(0, size_dim, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < size_dim
        
        # Load values
        x = tl.load(x_ptr + offs_m[:, None] * stride_other + offs_k[None, :] * stride_dim,
                   mask=mask_m[:, None] & mask_k[None, :],
                   other=float('-inf'))
        
        # Update max values and indices
        max_curr = tl.max(x, axis=1)
        idx_curr = tl.argmax(x, axis=1)
        
        # Update if new max is found
        update = max_curr > max_vals
        max_vals = tl.where(update, max_curr, max_vals)
        max_idxs = tl.where(update, k + idx_curr, max_idxs)
    
    # Store results
    tl.store(out_ptr + offs_m, max_vals, mask=mask_m)
    tl.store(idx_ptr + offs_m, max_idxs, mask=mask_m)

def max(x):
    """Compute maximum value in tensor using Triton kernels."""
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate intermediate and output tensors
    mid = torch.empty(n_blocks, device=x.device, dtype=x.dtype)
    out = torch.empty(1, device=x.device, dtype=x.dtype)
    
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
    )
    
    return out.item()

def max_dim(x, dim):
    """Compute maximum value along specified dimension."""
    # Validate dimension
    if dim >= x.ndim:
        raise ValueError(f"dim {dim} out of range for tensor of rank {x.ndim}")
    
    # Calculate sizes
    size_dim = x.size(dim)
    size_other = x.numel() // size_dim
    
    # Calculate strides
    stride_dim = x.stride(dim)
    stride_other = x.stride(0) if dim != 0 else x.stride(1)
    
    # Allocate output tensors
    out_vals = torch.empty(size_other, device=x.device, dtype=x.dtype)
    out_idxs = torch.empty(size_other, device=x.device, dtype=torch.int32)
    
    # Calculate grid dimensions
    BLOCK_SIZE_M = triton.next_power_of_2(min(size_other, 128))
    BLOCK_SIZE_K = triton.next_power_of_2(min(size_dim, 128))
    grid = (triton.cdiv(size_other, BLOCK_SIZE_M), 1)
    
    # Launch kernel
    max_kernel[grid](
        x.data_ptr(),
        out_vals.data_ptr(),
        out_idxs.data_ptr(),
        stride_dim,
        stride_other,
        size_dim,
        size_other,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape outputs to match input dimensions
    out_shape = list(x.shape)
    out_shape.pop(dim)
    return out_vals.reshape(out_shape), out_idxs.reshape(out_shape)
