import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_m,
    stride_weight,
    stride_output_batch,
    stride_output_m,
    N_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offset_x = pid_batch * stride_x_batch + pid_m * BLOCK_N_SIZE * stride_x_m
    offset_weight = pid_m * BLOCK_N_SIZE * stride_weight
    offset_output = pid_batch * stride_output_batch + pid_m * BLOCK_N_SIZE * stride_output_m

    var = tl.zeros([BLOCK_N_SIZE], dtype=tl.float32)
    for i in range(0, N_SIZE, BLOCK_N_SIZE):
        idx = i + tl.arange(0, BLOCK_N_SIZE)
        x = tl.load(x_ptr + offset_x + idx).to(tl.float32)
        var += x * x

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = 1 / tl.sqrt(var + eps)

    for i in range(0, N_SIZE, BLOCK_N_SIZE):
        idx = i + tl.arange(0, BLOCK_N_SIZE)
        weight = tl.load(rms_w_ptr + offset_weight + idx)
        x = tl.load(x_ptr + offset_x + idx).to(tl.float32)
        x_hat = x * rstd
        output = x_hat * weight
        tl.store(output_ptr + offset_output + idx, output)


def rmsnorm_triton_wrapper(x, weight, eps):
    batch, M, N = x.shape
    assert M % 16 == 0
    assert N == weight.shape[0]
    assert x.stride(1) == 1 and x.stride(2) == weight.stride(0)
    dtype = x.dtype
    if dtype == torch.float16:
        eps = 1e-5
    elif dtype == torch.bfloat16:
        eps = 1e-3
    output = torch.empty_like(x)
    N_SIZE = N
    rmsnorm_triton[(batch, M // 16)](
        x,
        weight,
        output,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        output.stride(0),
        output.stride(1),
        N_SIZE,
        eps,
        16,
    )
    return output
