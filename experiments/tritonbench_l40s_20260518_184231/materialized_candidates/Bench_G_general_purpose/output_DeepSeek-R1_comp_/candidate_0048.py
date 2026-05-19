import torch
import triton
import triton.language as tl

def dim_compress(inp, dims):
    all_dims = list(range(inp.dim()))
    keep_dims = [d for d in all_dims if d not in dims]
    permuted_dims = keep_dims + sorted(dims)
    permuted = inp.permute(*permuted_dims)
    M = 1
    for d in keep_dims:
        M *= inp.size(d)
    N = 1
    for d in sorted(dims):
        N *= inp.size(d)
    compressed = permuted.reshape(M, N)
    return compressed

@triton.jit
def mean_dim_kernel(
    X_ptr,
    Mean_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_mean_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)

    row_mask = off_m < M
    mean_accum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        cols = n + off_n
        col_mask = cols < N

        x_ptrs = X_ptr + (off_m[:, None] * stride_xm + cols[None, :] * stride_xn)
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        x = x.to(tl.float32)

        mean_accum += tl.sum(x, axis=1)

    mean = mean_accum / N

    mean_ptrs = Mean_ptr + off_m * stride_mean_m
    tl.store(mean_ptrs, mean, mask=row_mask)

def mean_dim(x, dim, keepdim=False):
    if dim is None:
        dim = list(range(x.dim()))
    if not dim:
        return x.clone()

    x_compressed = dim_compress(x, dim)
    M, N = x_compressed.shape
    out = torch.empty(M, device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 1024
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    mean_dim_kernel[grid](
        x_compressed, out, M, N,
        x_compressed.stride(0), x_compressed.stride(1),
        out.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    kept_dims = sorted(d for d in range(x.dim()) if d not in dim)
    kept_dims_shape = [x.size(d) for d in kept_dims]
    if keepdim:
        new_shape = []
        kept_idx = 0
        for d in range(x.dim()):
            if d in kept_dims:
                new_shape.append(kept_dims_shape[kept_idx])
                kept_idx += 1
            else:
                new_shape.append(1)
        out = out.view(new_shape)
    else:
        out = out.view(kept_dims_shape)
    
    return out
