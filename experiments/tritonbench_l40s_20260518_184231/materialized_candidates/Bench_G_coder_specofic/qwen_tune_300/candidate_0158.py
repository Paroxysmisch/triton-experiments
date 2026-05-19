import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_m,
    stride_x_n,
    stride_rms_w,
    stride_output_batch,
    stride_output_m,
    stride_output_n,
    N_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    x_m_ptr = x_ptr + pid_batch * stride_x_batch + pid_m * stride_x_m
    rms_w_n_ptr = rms_w_ptr + pid_m * stride_rms_w
    output_m_ptr = output_ptr + pid_batch * stride_output_batch + pid_m * stride_output_m

    # compute variance
    _var = tl.zeros([BLOCK_N_SIZE], dtype=tl.float32)
    for pid_n in range(tl.cdiv(N_SIZE, BLOCK_N_SIZE)):
        n_offsets = pid_n * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
        x_n_ptr = x_m_ptr + n_offsets * stride_x_n
        x = tl.load(x_n_ptr, mask=n_offsets < N_SIZE, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var, axis=0) / N_SIZE
    rstd = 1 / tl.math.sqrt(var + eps)

    # normalize, scale and write out
    for n in range(0, N_SIZE):
        x_n_ptr = x_m_ptr + n * stride_x_n
        rms_w_n_ptr = rms_w_ptr + n * stride_rms_w
        output_n_ptr = output_m_ptr + n * stride_output_n
        w = tl.load(rms_w_n_ptr)
        x = tl.load(x_n_ptr).to(tl.float32)
        x_hat = x * rstd
        output = x_hat * w
        tl.store(output_n_ptr, output)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    x = x.contiguous()
    rms_w = rms_w.contiguous()
    output = torch.empty_like(x)

    assert x.dim() == 3
    M, N = x.shape[-2:]
    BLOCK_N_SIZE = 32
    N_SIZE = triton.next_power_of_2(N)

    rmsnorm_triton[
        (
            x.shape[0],
            x.shape[1],
        )
    ](
        x,
        rms_w,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rms_w.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        N_SIZE=N_SIZE,
        eps=eps,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
    )

    return output
