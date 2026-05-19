import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   output_ptr: tl.pointer_type,
                   x_stride: tl.uint32,
                   rms_w_stride: tl.uint32,
                   output_stride: tl.uint32,
                   N_SIZE: tl.constexpr,
                   eps: tl.constexpr,
                   BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_N_SIZE)
    mask = offsets < N_SIZE

    x_row_start = x_ptr + pid_batch * x_stride[0] + pid_m * x_stride[1]
    x_ptrs = x_row_start + offsets

    rms_w_ptrs = rms_w_ptr + offsets

    x_row = tl.load(x_ptrs, mask=mask, other=0.0)

    sum_sq = tl.zeros((BLOCK_N_SIZE,), dtype=tl.float32)
    sum_sq += x_row * x_row

    for i in range(BLOCK_N_SIZE, N_SIZE, BLOCK_N_SIZE):
        x_ptrs = x_row_start + offsets + i
        x_row = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_sq += x_row * x_row

    var = sum_sq / N_SIZE
    rstd = tl.rsqrt(var + eps)

    x_row = tl.load(x_row_start + offsets, mask=mask, other=0.0)
    rms_w = tl.load(rms_w_ptrs, mask=mask, other=1.0)

    norm_x = x_row * rstd
    scaled_x = norm_x * rms_w

    output_ptrs = output_ptr + pid_batch * output_stride[0] + pid_m * output_stride[1] + offsets
    tl.store(output_ptrs, scaled_x, mask=mask)

def rmsnorm_triton_wrapper(x, rms_w, N_SIZE, eps=1e-5, BLOCK_N_SIZE=128):
    B, M, _ = x.shape

    assert x.is_cuda and rms_w.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

    output = torch.empty_like(x)

    grid = (B, M)
    rmsnorm_triton[grid](
        x, rms_w, output,
        x.stride(), rms_w.stride(), output.stride(),
        N_SIZE, eps, BLOCK_N_SIZE,
        num_warps=4, num_stages=1)

    return output

# Example usage
x = torch.randn(10, 20, 30, device='cuda', dtype=torch.float32)
rms_w = torch.ones(30, device='cuda', dtype=torch.float32)
output = rmsnorm_triton_wrapper(x, rms_w, N_SIZE=30)
print(output)
