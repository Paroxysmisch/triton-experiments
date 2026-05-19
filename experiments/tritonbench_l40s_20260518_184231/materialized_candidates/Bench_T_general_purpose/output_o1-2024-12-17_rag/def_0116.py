import torch
import triton
import triton.language as tl

@triton.jit
def sum_dim_kernel(X, Sum, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Sum = Sum + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        val = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += val

    s = tl.sum(_sum, axis=1)[:, None]
    tl.store(Sum, s, mask=row_mask)

def dim_compress(inp: torch.Tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

def sum(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    x = input
    if dim is None:
        # Reduce over all dimensions
        dim = list(range(x.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    dim = [d % x.ndim for d in dim]

    # Compress dimensions for reduction
    shape = list(x.shape)
    x = dim_compress(x, dim)

    # Compute reduction size N and batch size M
    N = 1
    for d in dim:
        N *= shape[d]
        shape[d] = 1
    M = x.numel() // N

    # Prepare output
    out = torch.empty(shape, dtype=dtype, device=x.device)

    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch.cuda.device(x.device):
        sum_dim_kernel[grid](x, out, M, N, BLOCK_M=8, BLOCK_N=8)

    # Squeeze if not keeping dimensions
    if not keepdim:
        for d in sorted(dim):
            out = out.squeeze(d)
    return out
