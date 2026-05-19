import torch
import triton
import triton.language as tl

def can_use_int32_index(tensor):
    return tensor.numel() <= torch.iinfo(torch.int32).max

def argmax(input, dim=None):
    assert input.is_contiguous(), "Input tensor must be contiguous"
    
    if dim is None:
        # Flatten the tensor and perform two-stage reduction
        inp_flat = input.view(-1)
        M = inp_flat.numel()
        int64_index = not can_use_int32_index(inp_flat)
        dtype = torch.int64 if int64_index else torch.int32
        
        BLOCK_SIZE = 4096  # Tunable based on hardware
        mid_size = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        mid_value = torch.empty((mid_size,), dtype=input.dtype, device=input.device)
        mid_index = torch.empty((mid_size,), dtype=dtype, device=input.device)
        
        grid = (mid_size,)
        argmax_kernel_1[grid](inp_flat, mid_value, mid_index, M, BLOCK_SIZE, int64_index)
        
        if mid_size == 1:
            return mid_index
        else:
            out = torch.empty((1,), dtype=dtype, device=input.device)
            BLOCK_MID = 512  # Tunable
            argmax_kernel_2[(1,)](mid_value, mid_index, out, mid_size, BLOCK_MID, int64_index)
            return out
    else:
        # Dimension-specific reduction
        assert dim < input.dim(), "Dimension out of range"
        shape = input.size()
        rank = input.dim()
        
        # Calculate problem dimensions
        M = 1
        for i in range(dim):
            M *= shape[i]
        N = shape[dim]
        K = 1
        for i in range(dim+1, rank):
            K *= shape[i]
        
        int64_index = not can_use_int32_index(input)
        dtype = torch.int64 if int64_index else torch.int32
        out = torch.empty((M, K), dtype=dtype, device=input.device)
        
        BLOCK_M = 128  # Tunable
        BLOCK_N = 4    # Tunable
        grid = (triton.cdiv(M, BLOCK_M), K)
        
        argmax_kernel[grid](
            input, out, M, N, K,
            input.stride(dim),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            int64_index=int64_index
        )
        return out.view(shape[:dim] + shape[dim+1:])

@triton.jit
def argmax_kernel_1(
    inp_ptr,
    mid_value_ptr,
    mid_index_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    
    # Load input block
    vals = tl.load(inp_ptr + offs, mask=mask, other=-float('inf'))
    idxs = tl.where(mask, offs, -1)
    if INT64_INDEX:
        idxs = idxs.to(tl.int64)
    
    # Reduce within block
    max_val, max_pos = tl.reduce((vals, idxs), 0, tl.argmax)
    
    # Store intermediate results
    tl.store(mid_value_ptr + pid, max_val)
    tl.store(mid_index_ptr + pid, max_pos)

@triton.jit
def argmax_kernel_2(
    mid_value_ptr,
    mid_index_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_MID) + pid * BLOCK_MID
    mask = offs < mid_size
    
    vals = tl.load(mid_value_ptr + offs, mask=mask, other=-float('inf'))
    idxs = tl.load(mid_index_ptr + offs, mask=mask, other=-1)
    if not INT64_INDEX:
        idxs = idxs.to(tl.int64)
    
    # Final reduction
    max_val, max_pos = tl.reduce((vals, idxs), 0, tl.argmax)
    tl.store(out_ptr, max_pos)

@triton.jit
def argmax_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    K,
    stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    int64_index: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize max values and indices
    max_vals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) - float('inf')
    max_idxs = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32 if not int64_index else tl.int64)
    
    for n in range(N):
        ptr = inp_ptr + off_m[:, None] * stride + n * stride + off_k[None, :]
        vals = tl.load(ptr, mask=(off_m[:, None] < M) & (off_k[None, :] < K), other=-float('inf'))
        
        # Update max
        new_max = vals > max_vals
        max_vals = tl.where(new_max, vals, max_vals)
        max_idxs = tl.where(new_max, n, max_idxs)
    
    # Store results
    out_ptrs = out_ptr + off_m[:, None] * K + off_k[None, :]
    tl.store(out_ptrs, max_idxs, mask=(off_m[:, None] < M) & (off_k[None, :] < K))
