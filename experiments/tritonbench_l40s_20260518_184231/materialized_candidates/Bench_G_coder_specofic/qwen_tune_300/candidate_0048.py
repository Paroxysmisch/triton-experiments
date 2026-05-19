import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union, List

@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        col_mask = n + cols < N
        x = tl.load(X + rows[:, None] * N + n + cols[None, :], row_mask & col_mask, eviction_policy="evict_last")
        _mean += x.to(tl.float32)

    mean = tl.sum(_mean, axis=1) / N
    tl.store(Mean + rows, mean[:, None], row_mask)

def dim_compress(inp: Tensor, dims: List[int]) -> Tensor:
    inp_shape = list(inp.shape)
    dims_order = sorted(dims)
    for dim in dims_order:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert len(dims) <= inp.ndim, "Invalid dims length"

    for i, dim in enumerate(dims):
        inp_shape.insert(dim + i + 1, 1)
    perm = [i for i in range(len(inp_shape))]
    inv_dims = [x for x in range(len(dims))][::-1]
    for i, dim in enumerate(dims):
        perm[dim + i + 1] = perm[dim]
        perm[dim] = inv_dims[i] + len(inp_shape) - len(dims)
    return inp.view(inp_shape).permute(perm)

def mean_dim(
    x: Union[Tensor, torch.Tensor],
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    return_indices: bool = False,
) -> Union[Tensor, torch.Tensor]:
    assert isinstance(x, (Tensor, torch.Tensor)), f"Invalid input type: {type(x)}"
    assert dim is not None, "Dimension must not be none"
    if isinstance(dim, int):
        dim = (dim,)
    assert all([d >= -x.ndim and d < x.ndim for d in dim]), "Invalid dim"
    assert len(dim) > 0 and len(dim) <= x.ndim, "Invalid dim length"

    x_shape = list(x.shape)
    dim = sorted(dim)
    x = dim_compress(x, dim)
    ndim = len(x.shape)
    M = 1
    N = 1
    for i in range(ndim):
        if i in dim:
            N *= x_shape[i]
        else:
            M *= x_shape[i]
    out_shape = list(x.shape)
    for i in dim:
        out_shape[i] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out
