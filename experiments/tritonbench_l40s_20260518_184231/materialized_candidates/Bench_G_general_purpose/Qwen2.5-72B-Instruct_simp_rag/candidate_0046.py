import torch
import triton
import triton.language as tl

# Triton kernel to compute the mean of a tensor along specified dimensions
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_end = min(row_start + BLOCK_M, M)
    row_range = tl.arange(row_start, row_end)

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        row_mask = row_range < M
        mask = row_mask[:, None] & col_mask[None, :]

        a = tl.load(X + row_range[:, None] * N + cols[None, :], mask, other=0.0).to(tl.float32)
        _mean += a

    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean + row_range[:, None], mean, row_mask[:, None])

# Helper function to compress specified dimensions for efficient processing
def dim_compress(inp: torch.Tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

# Main function to compute the mean along specified dimensions
def mean_dim(x, dim, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = torch.mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch.cuda.device(x.device):
        mean_dim_kernel[grid](x, out, M, N, BLOCK_M=8, BLOCK_N=8)
    if not keepdim:
        out = out.squeeze(dim)
    return out

# Example usage
b = torch.randn(2, 3, 4, 5, device="cuda")
print(mean_dim(b, [1, 2]))
