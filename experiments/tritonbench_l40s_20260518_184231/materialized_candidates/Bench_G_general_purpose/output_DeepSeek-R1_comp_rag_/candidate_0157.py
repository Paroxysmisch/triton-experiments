import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    output_ptr,
    x_stride_batch,
    x_stride_m,
    x_stride_n,
    rms_w_stride,
    output_stride_batch,
    output_stride_m,
    output_stride_n,
    N_SIZE,
    eps,
    BLOCK_N_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    x_row_ptr = x_ptr + pid_batch * x_stride_batch + pid_m * x_stride_m
    output_row_ptr = output_ptr + pid_batch * output_stride_batch + pid_m * output_stride_m
    
    sum_squares = 0.0
    for offset in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_N_SIZE)
        mask = col_offsets < N_SIZE
        x_vals = tl.load(x_row_ptr + col_offsets * x_stride_n, mask=mask, other=0.0)
        sum_squares += tl.sum(x_vals * x_vals)
    
    variance = sum_squares / N_SIZE
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    for offset in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_N_SIZE)
        mask = col_offsets < N_SIZE
        x_vals = tl.load(x_row_ptr + col_offsets * x_stride_n, mask=mask, other=0.0)
        w_vals = tl.load(rms_w_ptr + col_offsets * rms_w_stride, mask=mask, other=0.0)
        normalized = x_vals * rstd
        output = normalized * w_vals
        tl.store(output_row_ptr + col_offsets * output_stride_n, output, mask=mask)

def rmsnorm_triton_wrapper(x: torch.Tensor, rms_weight: torch.Tensor, eps: float = 1e-5):
    assert x.dim() == 3, "Input tensor must be 3D (batch, M, N)"
    B, M, N = x.shape
    assert rms_weight.shape == (N,), f"RMS weight must have shape ({N},)"
    
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert rms_weight.is_contiguous(), "RMS weight must be contiguous"
    assert x.is_cuda and rms_weight.is_cuda, "Inputs must be on CUDA"
    
    output = torch.empty_like(x)
    
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 2048:
        BLOCK_N = 2048
    
    grid = (B, M)
    rmsnorm_triton[grid](
        x, rms_weight, output,
        x.stride(0), x.stride(1), x.stride(2),
        rms_weight.stride(0),
        output.stride(0), output.stride(1), output.stride(2),
        N, eps, BLOCK_N,
        num_warps=8
    )
    return output

# Example usage:
# x = torch.randn(10, 20, 512, device='cuda')
# weight = torch.randn(512, device='cuda')
# output = rmsnorm_triton_wrapper(x, weight)
