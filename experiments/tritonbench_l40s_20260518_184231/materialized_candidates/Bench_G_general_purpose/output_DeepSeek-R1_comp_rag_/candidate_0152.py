import torch
import triton
import triton.language as tl
import math

# Helper function to check if 32-bit indices are sufficient
def can_use_int32_index(tensor):
    return tensor.numel() <= (1 << 31) - 1

# Kernel 1: Block-wise reduction for flat arrays
@triton.jit
def argmax_kernel_1(
    inp_ptr,
    mid_val_ptr,
    mid_idx_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    vals = tl.load(inp_ptr + offset, mask=mask, other=-float('inf'))
    max_val = tl.max(vals)
    
    # Find minimal index for max_val in block
    is_max = (vals == max_val) & mask
    indices = offset
    candidate_indices = tl.where(is_max, indices, M)
    min_idx = tl.min(candidate_indices, axis=0)
    
    tl.store(mid_val_ptr + pid, max_val)
    tl.store(mid_idx_ptr + pid, min_idx)

# Kernel 2: Final reduction from intermediate results
@triton.jit
def argmax_kernel_2(
    mid_val_ptr,
    mid_idx_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    vals = tl.load(mid_val_ptr + offset, mask=mask, other=-float('inf'))
    max_val = tl.max(vals)
    
    # Find minimal index among max candidates
    is_max = (vals == max_val) & mask
    candidate_indices = tl.load(mid_idx_ptr + offset, mask=is_max, other=mid_size)
    min_idx = tl.min(candidate_indices, axis=0)
    tl.store(out_ptr, min_idx)

# Kernel 3: Dimension-aware argmax
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128}, num_warps=4),
    ],
    key=['M', 'N']
)
@triton.jit
def argmax_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = row_idx < M

    max_val = tl.full((BLOCK_M, 1), -float('inf'), dtype=tl.float32)
    dtype = tl.int64 if INT64_INDEX else tl.int32
    max_idx = tl.zeros((BLOCK_M, 1), dtype=dtype)

    for col_off in range(0, N, BLOCK_N):
        col_idx = col_off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = col_idx < N
        mask = row_mask & col_mask

        vals = tl.load(inp_ptr + row_idx * N + col_idx, mask=mask, other=-float('inf'))
        indices = col_idx.to(dtype)

        larger = vals > max_val
        equal = vals == max_val
        smaller = indices < max_idx
        update = larger | (equal & smaller)

        max_val = tl.where(update, vals, max_val)
        max_idx = tl.where(update, indices, max_idx)

    tl.store(out_ptr + row_idx, max_idx, mask=row_mask)

# Wrapper function
def argmax(inp, dim=None, keepdim=False):
    if dim is None:
        # Flatten case
        M = inp.numel()
        use_int64 = not can_use_int32_index(inp)
        idx_dtype = torch.int64 if use_int64 else torch.int32
        
        BLOCK_SIZE = triton.next_power_of_2(min(1024, math.ceil(math.sqrt(M))))
        mid_size = triton.cdiv(M, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(mid_size)

        mid_val = torch.empty(mid_size, dtype=inp.dtype, device=inp.device)
        mid_idx = torch.empty(mid_size, dtype=idx_dtype, device=inp.device)
        
        out = torch.empty((), dtype=idx_dtype, device=inp.device) if not keepdim \
            else torch.full(inp.shape, 0, dtype=idx_dtype, device=inp.device)

        argmax_kernel_1[(mid_size,)](inp, mid_val, mid_idx, M, BLOCK_SIZE, use_int64)
        argmax_kernel_2[(1,)](mid_val, mid_idx, out, mid_size, BLOCK_MID, use_int64)
        
        return out if not keepdim else out.reshape([1]*inp.ndim)
    else:
        # Dimension reduction case
        dim = [dim] if isinstance(dim, int) else dim
        org_dim = [d % inp.ndim for d in dim]
        org_shape = inp.shape
        
        # Compress reduction dimensions
        kept_dims = [i for i in range(inp.ndim) if i not in org_dim]
        compressed = inp.permute(*kept_dims, *org_dim).flatten(0, len(kept_dims)-1)
        if compressed.shape[:-1] == (0,):  # Handle empty tensors
            return torch.zeros(compressed.shape[:-1], dtype=torch.int64, device=inp.device)
        
        M, N = compressed.shape[0], compressed.shape[1]
        use_int64 = not can_use_int32_index(compressed)
        idx_dtype = torch.int64 if use_int64 else torch.int32
        
        out = torch.empty(M, dtype=idx_dtype, device=inp.device)
        
        def grid(meta): return (triton.cdiv(M, meta['BLOCK_M']),)
        argmax_kernel[grid](compressed, out, M, N, BLOCK_M=1, BLOCK_N=1024, INT64_INDEX=use_int64)
        
        # Restore original dimensions
        out_shape = list(org_shape)
        for d in org_dim:
            out_shape[d] = 1
        out = out.view(out_shape)
        return out if keepdim else out.squeeze(org_dim)
