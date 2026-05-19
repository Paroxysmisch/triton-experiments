import torch
import triton
import triton.language as tl

@triton.jit
def sum_dim_kernel(X, Sum, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Sum = Sum + pid
    row_mask = pid < M

    # Compute sum
    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a
    sum_val = tl.sum(_sum, axis=1)
    sum_val = sum_val[:, None]
    tl.store(Sum, sum_val, row_mask)

def dim_compress(inp: torch.Tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

def sum(input, dim, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    if dim is None:
        out = input.sum(dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * input.ndim)
        return out

    shape = list(input.shape)
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % input.ndim for d in dim]
    input = dim_compress(input, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = input.numel() // N
    out_shape = shape if keepdim else [s for i, s in enumerate(shape) if i not in dim]
    out = torch.empty(out_shape, dtype=dtype, device=input.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch.cuda.device(input.device):
        sum_dim_kernel[grid](input, out, M, N, BLOCK_M=8, BLOCK_N=8)
    return out

# Example usage
b = torch.randn(2, 3, 4, 5, device="cuda")
print(sum(b, [1, 2]))
