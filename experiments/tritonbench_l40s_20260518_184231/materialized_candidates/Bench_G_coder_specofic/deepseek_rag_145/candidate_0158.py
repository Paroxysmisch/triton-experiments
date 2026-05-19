import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr: tl.pointer(tl.float32),
    rms_w_ptr: tl.pointer(tl.float32),
    output_ptr: tl.pointer(tl.float32),
    x_row_stride: tl.constexpr,
    rms_w_stride: tl.constexpr,
    output_stride: tl.constexpr,
    N_SIZE: tl.constexpr,
    eps: tl.float32,
    BLOCK_N_SIZE: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr
):
    pid_m = tl.program_id(1)
    pid_batch = tl.program_id(0)

    batch_start_ptr = pid_batch * x_row_stride
    batch_x_ptr = x_ptr + batch_start_ptr
    batch_rms_w_ptr = rms_w_ptr + pid_m * rms_w_stride
    batch_output_ptr = output_ptr + batch_start_ptr

    offsets = tl.arange(0, N_SIZE)

    mask = offsets < N_SIZE
    x_chunk = tl.load(batch_x_ptr + offsets, mask=mask, other=0)
    rms_w = tl.load(batch_rms_w_ptr, mask=mask, other=1.0)

    x_square_chunk = x_chunk * x_chunk
    norm_square = tl.sum(x_square_chunk) / N_SIZE
    rstd = tl.rsqrt(norm_square + eps)

    output_chunk = x_chunk * rstd * rms_w
    tl.store(batch_output_ptr + offsets, output_chunk, mask=mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-9, BLOCK_N_SIZE=256, BLOCK_M_SIZE=16):
    N, M = x.shape[-2], x.shape[-1]
    output = torch.empty_like(x)

    grid = (triton.next_power_of_2(M) // BLOCK_M_SIZE, triton.next_power_of_2(N) // BLOCK_N_SIZE)
    rmsnorm_triton[grid](x, rms_w, output, torch.as_tensor(x.stride(-2)), torch.as_tensor(rms_w.stride(0)), torch.as_tensor(output.stride(-2)), N, eps, BLOCK_N_SIZE, BLOCK_M_SIZE)

    return output
