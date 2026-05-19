import torch
import triton
import triton.language as tl

@triton.jit
def mean_dim_kernel(
    X, 
    Mean, 
    M, 
    N, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    col_idx = 0
    while col_idx < N:
        col_offsets = col_idx + tl.arange(0, BLOCK_N)
        mask = (row_offsets < M) & (col_offsets < N)
        row_ptr = X + row_offsets * N
        x_ptr = row_ptr + col_offsets
        x = tl.load(x_ptr, mask=mask, other=0.0)
        # Each "x" is shape [BLOCK_M]. Sum into "acc"
        acc += x
        col_idx += BLOCK_N

    acc = acc / tl.float32(N)
    tl.store(Mean + row_offsets, acc, mask=(row_offsets < M)) 


def dim_compress(inp: torch.Tensor, dims: list):
    dims = sorted(dims)
    ndims = list(range(inp.ndim))
    for d in reversed(dims):
        ndims.pop(d)
    new_order = ndims + dims
    permuted = inp.permute(new_order)
    # Split shape into (rest, reduce)
    rest_shape = permuted.shape[:-len(dims)]
    reduce_shape = permuted.shape[-len(dims):]
    permuted = permuted.reshape(-1, torch.prod(torch.tensor(reduce_shape)))
    return permuted, rest_shape, reduce_shape


def mean_dim(x: torch.Tensor, dim: list, keepdim: bool = False):
    if isinstance(dim, int):
        dim = [dim]
    permuted, rest_shape, reduce_shape = dim_compress(x, dim)
    M = permuted.shape[0]
    N = permuted.shape[1]
    out_shape = list(rest_shape)
    
    out = torch.empty((M,), dtype=permuted.dtype, device=permuted.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    grid = lambda meta: (triton.cdiv(M, BLOCK_M),)

    mean_dim_kernel[grid](
        permuted, 
        out, 
        M, 
        N, 
        BLOCK_M, 
        BLOCK_N
    )
    
    out = out.reshape(out_shape)
    if keepdim:
        for d in sorted(dim):
            out = out.unsqueeze(d)
    return out
