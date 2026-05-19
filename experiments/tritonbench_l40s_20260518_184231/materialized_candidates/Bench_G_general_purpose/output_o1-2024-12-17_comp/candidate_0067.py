import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X, W, Y,
    M, N,
    stride_xm, stride_xn,
    stride_w,
    stride_ym, stride_yn,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each block processes one row
    row_idx = pid
    block_start = 0
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load row elements
    x_ptrs = X + row_idx * stride_xm + offsets * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute variance
    var = tl.sum(x * x, axis=0) / N
    # Compute reciprocal of RMS
    rrms = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale by weight
    w_ptrs = W + offsets * stride_w
    w = tl.load(w_ptrs, mask=mask, other=1.0)
    y = (x * rrms).to(tl.float32) * w  # Intermediate in float32 if needed

    # Store results
    y_ptrs = Y + row_idx * stride_ym + offsets * stride_yn
    tl.store(y_ptrs, y.to(tl.load(Y, mask=False).dtype), mask=mask)

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        M, N = x.shape
        y = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (M,)
        rms_norm_kernel[grid](
            x, w, y,
            M, N,
            x.stride(0), x.stride(1),
            w.stride(0),
            y.stride(0), y.stride(1),
            eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return y

def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, weight, eps)
