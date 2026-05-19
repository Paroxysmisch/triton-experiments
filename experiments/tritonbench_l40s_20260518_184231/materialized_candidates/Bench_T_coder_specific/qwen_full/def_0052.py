import torch
import triton
import triton.language as tl

@triton.jit
def sum_std_kernel(
    input,
    mean,
    sum2,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < N
    input = tl.load(input + pid, mask=mask, other=0.0)
    if N != M:
        mean = tl.load(mean + pid, mask=mask, other=0.0)
    mean = tl.sum(input) / M
    input = tl.where(mask, input - mean, 0.0)
    sum2 = tl.sum(input * input)
    return mean, sum2, N

def sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    if correction < 0:
        raise ValueError("correction must be greater than or equal to 0")

    if out is None:
        out = torch.empty(input.shape, dtype=dtype, device=input.device)
    else:
        out.copy_(input)
    if dim is None:
        input = input.ravel()
        dim = 0
    else:
        input = input.contiguous()
    out = out.contiguous()
    if correction > 0:
        mean = torch.empty_like(out, dtype=dtype)
        sum2 = torch.empty_like(out, dtype=dtype)
    else:
        mean = None
        sum2 = out
    N = input.numel()
    M = out.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(N)))
    grid = (math.ceil(math.sqrt(M)),)
    sum_std_kernel[grid](
        input,
        mean,
        sum2,
        N,
        M,
        BLOCK_SIZE=block_size,
    )
    if correction > 0:
        out.mul_(N)
        mean.mul_(N)
        sum2.mul_(N)
        mask = tl.arange(0, block_size) < M
        var = tl.where(mask, sum2 - mean * mean, 0.0) / (max(0, M - correction))
        out = torch.sqrt(var, out=out)
        if not keepdim:
            out = out.reshape(input.shape)
    return out
