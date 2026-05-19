import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr
):
    # Triton kernel for L2 normalization
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x_ptrs = X + row * stride_x_row + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = x * rstd
    y_ptrs = Y + row * stride_x_row + cols
    tl.store(y_ptrs, y, mask=cols < N)

def _l2_norm_fwd(x, eps=1e-5):
    # Function to call the Triton kernel
    M, N = x.shape
    x = x.reshape(-1, N)
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(x.element_size() // 8)
    if BLOCK_N > 65536:
        raise RuntimeError("This layer norm doesn't support column dim >= 64KB.")
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 256.")
    _l2_norm_fwd_1pass_kernel[(M,)](
        x, y,
        x.stride(0), N, eps,
        BLOCK_N=BLOCK_N
    )
    return y.reshape_as(x)
