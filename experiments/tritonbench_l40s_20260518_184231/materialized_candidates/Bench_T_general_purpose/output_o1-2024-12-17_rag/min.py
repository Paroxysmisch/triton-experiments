import torch
import triton
import triton.language as tl

@triton.jit
def _min_reduce_kernel(
    input_ptr,
    output_val_ptr,
    output_idx_ptr,
    stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_offset = row_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    min_val = tl.float32(float('inf'))
    min_idx = tl.int32(-1)
    
    for start_col in range(0, n_cols, BLOCK_SIZE):
        cur_cols = start_col + col_offsets
        mask = cur_cols < n_cols
        vals = tl.load(input_ptr + row_offset + cur_cols, mask=mask, other=float('inf'))
        is_less = vals < min_val
        min_val = tl.where(is_less, vals, min_val)
        min_idx = tl.where(is_less, cur_cols, min_idx)

    tl.store(output_val_ptr + row_id, min_val)
    tl.store(output_idx_ptr + row_id, tl.cast(min_idx, tl.int64))

def min(input, dim, keepdim=False, *, out=None):
    dim = dim if dim >= 0 else dim + input.dim()
    perm = list(range(input.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = input.permute(perm)
    N = x_perm[..., 0].numel()
    M = x_perm.shape[-1]
    x_reshaped = x_perm.reshape(N
