import triton
import triton.language as tl
import torch

@triton.jit
def mean_dim_kernel(
    X,
    Mean,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_mask = pid < M

    col_offsets = tl.arange(0, BLOCK_N)
    x_ptrs = X + pid * BLOCK_N + col_offsets
    x_mask = col_offsets < N

    _mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    _cnt = tl.zeros([BLOCK_N], dtype=tl.float32)
    for block_start in range(0, N, BLOCK_N):
        x = tl.load(x_ptrs, mask=x_mask & row_mask, other=0.0).to(tl.float32)
        _mean += x
        _cnt += 1

        x_ptrs += BLOCK_M * BLOCK_N

    mean = tl.sum(_mean, axis=0) / tl.sum(_cnt, axis=0)
    if row_mask:
        tl.store(Mean + pid, mean)

def dim_compress(inp, dims):
    dims = list(set(inp.ndim - 1 - i for i in dims))
    inp_perm = inp.permute(dims + [i for i in range(inp.ndim - 1) if i not in dims])
    return inp_perm

def mean_dim(x, dim, keepdim=False, *, dtype=None):
    x = dim_compress(x, dim)
    M = torch.prod(torch.tensor(x.shape[:-1])).item()
    N = torch.prod(torch.tensor(x.shape[-1:])).item()
    x = x.contiguous()
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    mean_dim_kernel[grid](x, out, M, N)

    shape = list(x.shape[:-1]) + [N]
    if not keepdim:
        out = out.reshape(shape)
        out = out.squeeze(dim=list(set(range(out.ndim - 1)) - set(dim)))

    return out
