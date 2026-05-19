import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def batch_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    output_ptr,
    n,
    C,
    HW,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid * BLOCK_N

    rmean_ptr = running_mean_ptr + block_id
    rvar_ptr = running_var_ptr + block_id

    mean = 0.0
    var = 0.0
    for i in range(0, BLOCK_N):
        for j in range(0, BLOCK_C):
            offset = (block_id + i) * C * HW + j * HW + tl.arange(0, HW)
            mask = offset < n * HW
            x = tl.load(input_ptr + offset, mask=mask).to(tl.float32)
            mean += x
            var += x * x

    mean /= n * HW
    xhat = 0.0
    for i in range(0, BLOCK_N):
        for j in range(0, BLOCK_C):
            offset = (block_id + i) * C * HW + j * HW + tl.arange(0, HW)
            mask = offset < n * HW
            x = tl.load(input_ptr + offset, mask=mask).to(tl.float32)
            xhat += (x - mean)

    xhat /= n * HW
    var = var / n - xhat * xhat
    rsqrt_var = tl.rsqrt(var + eps)

    for i in range(0, BLOCK_N):
        for j in range(0, BLOCK_C):
            offset = (block_id + i) * C * HW + j * HW + tl.arange(0, HW)
            mask = offset < n * HW
            x = tl.load(input_ptr + offset, mask=mask).to(tl.float32)
            x_hat = (x - mean) * rsqrt_var
            if weight_ptr is not None:
                w = tl.load(weight_ptr + offset, mask=mask).to(tl.float32)
                x_hat = x_hat * w
            if bias_ptr is not None:
                b = tl.load(bias_ptr + offset, mask=mask).to(tl.float32)
                x_hat = x_hat + b
            tl.store(output_ptr + offset, x_hat.to(output_ptr.dtype.element_ty), mask=mask)

    tl.store(rmean_ptr, mean)
    tl.store(rvar_ptr, var)

def batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor = None,
    bias: Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tensor:
    assert input.is_contiguous()
    assert running_mean.is_contiguous()
    assert running_var.is_contiguous()
    assert input.shape == running_mean.shape
    assert input.shape == running_var.shape
    if weight is not None:
        assert weight.is_contiguous()
        assert input.shape == weight.shape
    if bias is not None:
        assert bias.is_contiguous()
        assert input.shape == bias.shape

    n = input.numel() / input.size(-1)
    output = torch.empty_like(input)

    BLOCK_N = 8
    BLOCK_C = 16
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"] * meta["BLOCK_C"]),)

    with torch.cuda.device(input.device):
        batch_norm_kernel[grid](
            input,
            weight,
            bias,
            running_mean,
            running_var,
            output,
            n,
            input.size(-1),
            input.numel() // input.size(-1),
            BLOCK_N=BLOCK_N,
            BLOCK_C=BLOCK_C,
            eps=eps,
        )
    return output
