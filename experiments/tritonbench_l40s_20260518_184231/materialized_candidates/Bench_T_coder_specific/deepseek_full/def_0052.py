import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def sum_std_kernel(
    input,
    dim,
    keepdim,
    dtype,
    correction,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input + offset, mask=mask)
    x = tl.where(dtype == tl.int32, x.to(tl.float32), x)
    sum_ = tl.sum(x, axis=0)
    mean = sum_ / tl.maximum(0, N - correction * N)
    x_mean = tl.where(mask, x - mean, 0)
    var = tl.sum(x_mean * x_mean, axis=0) / tl.maximum(0, N - correction * N)
    std = tl.sqrt(var)
    if not keepdim:
        std = tl.where(dim, std, 1)
    tl.store(out + offset, std, mask=mask)


def sum_std(
    input: Tensor,
    dim=None,
    keepdim=False,
    dtype=None,
    correction=1,
    out=None,
) -> Tensor:
    func_inputs = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype,
        "correction": correction,
        "out": out,
    }
    triton_dtype = get_triton_dtype(dtype)
    input, dim, keepdim, out, shape, dist_spec, broadcast_dims = broadcast_shapes_dims(
        input, dim, keepdim, out
    )
    N = input.numel() if dim is None else input.shape[dim]
    if out is None:
        out = torch.empty(shape, dtype=triton_dtype, device=input.device)
    max_grid = (triton.cdiv(N, 128),)
    with torch.cuda.device(input.device.index):
        sum_std_kernel[max_grid](
            input,
            dim,
            keepdim,
            triton_dtype,
            correction,
            out,
            N,
            BLOCK_SIZE=128,
        )
    return out
