import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Calculate offset for the m dimension
    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)

    # Compute variance
    for block_n_start_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_start_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)
        xf = x.to(tl.float32)
        var += xf * xf

    var = tl.sum(var, axis=0) / N_SIZE
    std = tl.sqrt(var + eps)

    # Normalize and apply RMS weights
    for block_n_start_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_start_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE

        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)

        x_new = x / std
        out = x_new * rms_w_offset
        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)

def rmsnorm_wrapper(x, rms_weights, eps=1e-6):
    batch, M, K = x.shape
    out = torch.empty_like(x)
    rmsnorm_triton[(batch, M,)](x, rms_weights, out,
                                *x.stride(),
                                *rms_weights.stride(),
                                *out.stride(),
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=4096,
                                num_warps=16)
    return out

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rms_weights, eps=1e-6):
        ctx.save_for_backward(x, rms_weights)
        ctx.eps = eps
        return rmsnorm_wrapper(x, rms_weights, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x, rms_weights = ctx.saved_tensors
        eps = ctx.eps
        # Implement backward pass logic here if needed
        # This is a placeholder for gradient calculation
        grad_x = grad_output.clone()
        grad_rms_weights = grad_output.clone()
        return grad_x, grad_rms_weights, None

# Usage example
x = torch.randn(32, 128, 512, device='cuda', dtype=torch.float16)
rms_weights = torch.randn(512, device='cuda', dtype=torch.float16)
output = RmsNorm.apply(x, rms_weights)
