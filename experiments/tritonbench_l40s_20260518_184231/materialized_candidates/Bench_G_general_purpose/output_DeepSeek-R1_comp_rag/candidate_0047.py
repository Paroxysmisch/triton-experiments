import torch
import triton
import triton.language as tl

@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_start = X + pid * N
    mean_ptr = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    tl.store(mean_ptr, mean[:, None], mask=row_mask)

def dim_compress(inp: torch.Tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

def mean_dim(x, dim, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = x.mean(dtype=dtype)
        if keepdim:
            out = out.reshape([1] * x.ndim)
        return out
    
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    
    original_ndim = x.ndim
    num_reduction_dims = len(dim)
    reduction_dims = list(range(original_ndim - num_reduction_dims, original_ndim))
    
    N = 1
    for i in reduction_dims:
        N *= x.shape[i]
    M = x.numel() // N
    
    shape = list(x.shape)
    for i in reduction_dims:
        shape[i] = 1
    out = torch.empty(shape, dtype=dtype, device=x.device)
    
    BLOCK_M = 8
    BLOCK_N = 8
    grid = (triton.cdiv(M, BLOCK_M),)
    
    mean_dim_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    if not keepdim:
        squeeze_dims = [d + x.ndim - original_ndim for d in reduction_dims]
        out = out.squeeze(squeeze_dims)
    return out

# Test
b = torch.randn(2, 3, 4, 5, device="cuda")
print(mean_dim(b, [1, 2]))
