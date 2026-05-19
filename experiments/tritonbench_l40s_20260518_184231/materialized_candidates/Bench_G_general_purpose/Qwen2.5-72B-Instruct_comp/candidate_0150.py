import triton
import triton.language as tl

# Helper function to determine if int32 indices can be used
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def can_use_int32_index(M, N, K):
    return M * N * K < 2**31

# Kernel for the first stage of reduction when dim=None
@triton.jit
def argmax_kernel_1(inp, mid_value, mid_index, M, BLOCK_SIZE, INT64_INDEX):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, M)
    
    if INT64_INDEX:
        index_type = tl.int64
    else:
        index_type = tl.int32
    
    max_val = -float('inf')
    max_idx = index_type(-1)
    
    for i in range(block_start, block_end):
        val = tl.load(inp + i)
        if val > max_val:
            max_val = val
            max_idx = i
    
    tl.store(mid_value + pid, max_val)
    tl.store(mid_index + pid, max_idx)

# Kernel for the second stage of reduction when dim=None
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID, INT64_INDEX):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_MID
    block_end = min(block_start + BLOCK_MID, mid_size)
    
    if INT64_INDEX:
        index_type = tl.int64
    else:
        index_type = tl.int32
    
    max_val = -float('inf')
    max_idx = index_type(-1)
    
    for i in range(block_start, block_end):
        val = tl.load(mid_value + i)
        idx = tl.load(mid_index + i)
        if val > max_val:
            max_val = val
            max_idx = idx
    
    tl.store(out, max_idx)

# Kernel for argmax when a specific dimension is specified
@triton.jit
def argmax_kernel(inp, out, M, N, K, BLOCK_M, BLOCK_N, dim, INT64_INDEX):
    pid = tl.program_id(axis=0)
    block_start_m = (pid // (N // BLOCK_N)) * BLOCK_M
    block_start_n = (pid % (N // BLOCK_N)) * BLOCK_N
    
    if INT64_INDEX:
        index_type = tl.int64
    else:
        index_type = tl.int32
    
    if dim == 0:
        max_val = -float('inf')
        max_idx = index_type(-1)
        
        for i in range(block_start_m, min(block_start_m + BLOCK_M, M)):
            for j in range(block_start_n, min(block_start_n + BLOCK_N, N)):
                val = tl.load(inp + i * K + j)
                if val > max_val:
                    max_val = val
                    max_idx = i
        
        tl.store(out + block_start_n + pid % (N // BLOCK_N), max_idx)
    
    elif dim == 1:
        max_val = -float('inf')
        max_idx = index_type(-1)
        
        for i in range(block_start_m, min(block_start_m + BLOCK_M, M)):
            for j in range(block_start_n, min(block_start_n + BLOCK_N, N)):
                val = tl.load(inp + i * K + j)
                if val > max_val:
                    max_val = val
                    max_idx = j
        
        tl.store(out + block_start_m + pid // (N // BLOCK_N), max_idx)

import torch

def argmax(input_tensor, dim=None):
    if dim is None:
        M = input_tensor.numel()
        INT64_INDEX = can_use_int32_index(M, 1, 1) == 0
        BLOCK_SIZE = 1024
        mid_size = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        mid_value = torch.empty(mid_size, device=input_tensor.device, dtype=input_tensor.dtype)
        mid_index = torch.empty(mid_size, device=input_tensor.device, dtype=torch.int64 if INT64_INDEX else torch.int32)
        out = torch.empty(1, device=input_tensor.device, dtype=torch.int64 if INT64_INDEX else torch.int32)
        
        argmax_kernel_1[(mid_size,)](input_tensor, mid_value, mid_index, M, BLOCK_SIZE, INT64_INDEX)
        argmax_kernel_2[(1,)](mid_value, mid_index, out, mid_size, BLOCK_SIZE, INT64_INDEX)
        
        return out.item()
    
    else:
        shape = input_tensor.shape
        M = shape[dim]
        N = shape[dim - 1] if dim > 0 else 1
        K = shape[dim + 1] if dim < len(shape) - 1 else 1
        INT64_INDEX = can_use_int32_index(M, N, K) == 0
        BLOCK_M = 16
        BLOCK_N = 16
        grid = (N * M // (BLOCK_M * BLOCK_N),)
        out = torch.empty((N, M), device=input_tensor.device, dtype=torch.int64 if INT64_INDEX else torch.int32)
        
        argmax_kernel[grid](input_tensor, out, M, N, K, BLOCK_M, BLOCK_N, dim, INT64_INDEX)
        
        return out
