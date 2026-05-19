import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_x_b, stride_x_m, stride_x_k,
    stride_w,
    stride_y_b, stride_y_m, stride_y_k,
    N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offset_m = pid_b * stride_x_b + pid_m * stride_x_m
    n_idxs = tl.arange(0, BLOCK_SIZE)
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute variance across columns
    for start_n in range(0, N_SIZE, BLOCK_SIZE):
        offset_n = start_n + n_idxs
        mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        var_acc += x_fp32 * x_fp32

    var = tl.sum(var_acc, axis=0) / N_SIZE
    inv_rms = 1.0 / tl.sqrt(var + eps)

    # Apply normalization and store
    for start_n in range(0, N_SIZE, BLOCK_SIZE):
        offset_n = start_n + n_idxs
        mask = offset_n < N_SIZE

        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=mask, other=0.0)
        w = tl.load(w_ptr + offset_n * stride_w, mask=mask, other=1.0)

        out = (x.to(tl.float32) * inv_rms) * w
        out_offset = pid_b * stride_y_b + pid_m * stride_y_m + offset_n * stride_y_k
        tl.store(y_ptr + out_offset, out, mask=mask)


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        batch, m, k = x.shape
        y = torch.empty_like(x)
        grid = (batch, m)
        rms_norm_kernel[grid](
            x, w, y,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0),
            y.stride(0), y.stride(1), y.stride(2),
            N_SIZE=k,
            eps=eps,
            BLOCK_SIZE=4096,
            num_warps=4
        )
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        # For brevity, this backward pass can be implemented as needed
        raise NotImplementedError("Backward pass not implemented")

def rms_norm(x, normalized_shape, w, eps=1e-6):
    return RmsNorm.apply(x, w, eps)
