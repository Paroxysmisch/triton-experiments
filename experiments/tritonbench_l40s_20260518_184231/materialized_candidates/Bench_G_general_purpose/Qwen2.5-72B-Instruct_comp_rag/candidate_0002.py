import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_fwd_kernel(
    X, Y, OUT,
    stride_xm, stride_ym, stride_xn, stride_yn, stride_on,
    M, N,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_N

    offsets = block_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N

    x_ptrs = X + tl.arange(0, M)[:, None] * stride_xm + offsets[None, :] * stride_xn
    y_ptrs = Y + tl.arange(0, M)[:, None] * stride_ym + offsets[None, :] * stride_yn
    out_ptrs = OUT + tl.arange(0, M)[:, None] * stride_xm + offsets[None, :] * stride_on

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)

    sig_x = 1 / (1 + tl.exp(-x))
    out = sig_x * y

    tl.store(out_ptrs, out, mask=mask)

def _swiglu_fwd(x, y):
    M, N = x.shape
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Ensure the input tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Initialize the output tensor
    out = torch.empty_like(x)

    # Define the grid configuration
    grid = (triton.cdiv(N, 128),)

    # Launch the kernel
    _swiglu_fwd_kernel[grid](
        x, y, out,
        x.stride(0), y.stride(0), x.stride(1), y.stride(1), out.stride(1),
        M, N,
        BLOCK_SIZE_N=128
    )

    return out
