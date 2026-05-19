import torch
import triton
import triton.language as tl
import math

@triton.jit
def argmax_kernel_1(
    inp_ptr,
    mid_value_ptr,
    mid_index_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    # Get program ID and compute offset
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    
    # Compute offsets and load data
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    
    # Load input values
    x = tl.load(inp_ptr + offset, mask=mask, other=-float('inf'))
    
    # Find max value and its index within this block
    max_val = -float('inf')
    max_idx = 0
    
    for i in range(BLOCK_SIZE):
        if x[i] > max_val:
            max_val = x[i]
            max_idx = pid * BLOCK_SIZE + i
    
    # Store intermediate results
    tl.store(mid_value_ptr + pid, max_val)
    tl.store(mid_index_ptr + pid, max_idx)

@triton.jit
def argmax_kernel_2(
    mid_value_ptr,
    mid_index_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    # Load intermediate values and indices
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    values = tl.load(mid_value_ptr + offset, mask=mask, other=-float('inf'))
    indices = tl.load(mid_index_ptr + offset, mask=mask, other=0)
    
    # Find global maximum
    max_val = -float('inf')
    max_idx = 0
    
    for i in range(BLOCK_MID):
        if values[i] > max_val:
            max_val = values[i]
            max_idx = indices[i]
    
    # Store final result
    tl.store(out_ptr, max_idx)

@triton.jit
def argmax_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_im,
    stride_in,
    stride_ik,
    stride_om,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    # Calculate indices for this block
    m_idx = pid // K
    k_idx = pid % K
    
    # Initialize max tracking
    max_val = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    max_idx = tl.zeros([BLOCK_M], dtype=tl.int64 if INT64_INDEX else tl.int32)
    
    # Load and process input values
    for n in range(0, N, BLOCK_N):
        # Calculate input pointer
        offset_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offset_n = n + tl.arange(0, BLOCK_N)
        
        # Load mask
        m_mask = offset_m < M
        n_mask = offset_n < N
        mask = m_mask[:, None] & n_mask[None, :]
        
        # Load values
        block_ptr = inp_ptr + offset_m[:, None] * stride_im + offset_n[None, :] * stride_in + k_idx * stride_ik
        x = tl.load(block_ptr, mask=mask, other=-float('inf'))
        
        # Update max values and indices
        for i in range(BLOCK_N):
            max_val_new = tl.maximum(max_val, x[:, i])
            max_idx = tl.where(max_val_new > max_val, n + i, max_idx)
            max_val = max_val_new
    
    # Store results
    out_ptr = out_ptr + m_idx * BLOCK_M * stride_om + k_idx * stride_ok
    tl.store(out_ptr, max_idx, mask=m_mask)

def argmax(input_tensor, dim=None, keepdim=False):
    if dim is None:
        # Flatten case
        M = input_tensor.numel()
        BLOCK_SIZE = triton.next_power_of_2(min(M, 1024))
        n_blocks = triton.cdiv(M, BLOCK_SIZE)
        
        # Allocate intermediate buffers
        device = input_tensor.device
        mid_values = torch.empty(n_blocks, dtype=input_tensor.dtype, device=device)
        mid_indices = torch.empty(n_blocks, dtype=torch.int64, device=device)
        
        # Use 64-bit indexing for large tensors
        use_int64 = M >= (2**31)
        
        # Launch kernels
        argmax_kernel_1[(n_blocks,)](
            input_tensor, mid_values, mid_indices,
            M, BLOCK_SIZE=BLOCK_SIZE, INT64_INDEX=use_int64
        )
        
        out = torch.empty((), dtype=torch.int64, device=device)
        argmax_kernel_2[(1,)](
            mid_values, mid_indices, out,
            n_blocks, BLOCK_MID=triton.next_power_of_2(n_blocks)
        )
        
        return out.reshape(1) if keepdim else out
    
    else:
        # Handle negative dimensions
        dim = dim if dim >= 0 else input_tensor.dim() + dim
        
        # Calculate shapes
        shape = list(input_tensor.shape)
        M = 1
        for i in range(0, dim):
            M *= shape[i]
        N = shape[dim]
        K = 1
        for i in range(dim + 1, len(shape)):
            K *= shape[i]
            
        # Calculate strides
        strides = list(input_tensor.stride())
        stride_im = strides[0] if dim > 0 else 0
        stride_in = strides[dim]
        stride_ik = strides[-1] if dim < len(shape)-1 else 0
        
        # Output shape and strides
        out_shape = shape[:dim] + [1 if keepdim else None] + shape[dim+1:]
        out_shape = [s for s in out_shape if s is not None]
        out = torch.empty(out_shape, dtype=torch.int64, device=input_tensor.device)
        
        stride_om = M if K > 1 else 1
        stride_ok = 1
        
        # Launch kernel
        grid = (M * K,)
        argmax_kernel[grid](
            input_tensor, out,
            M, N, K,
            stride_im, stride_in, stride_ik,
            stride_om, stride_ok,
            BLOCK_M=min(M, 32),
            BLOCK_N=min(N, 128),
            INT64_INDEX=(M*N*K >= 2**31)
        )
        
        return out
