import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, stride_y_row, N, eps, BLOCK_N: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride_y_row
    X += row * stride_x_row

    # Compute variance
    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Normalize and write output
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0)
        y = x * rstd
        tl.store(Y + cols, y, mask=mask)

def _l2_norm_fwd(x, eps):
    # Reshape input for processing
    x_arg = x.reshape(-1, x.shape[-1])
    if not x_arg.is_contiguous():
        x_arg = x_arg.contiguous()

    y = torch.empty_like(x_arg)
    M, N = x_arg.shape

    # Determine the block size for Triton kernel
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 65536 // x.element_size():
        BLOCK_N = 65536 // x.element_size()
    BLOCK_N = max(BLOCK_N, 128)

    # Launch Triton kernel
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    _l2_norm_fwd_1pass_kernel[(M,)](
        x_arg, y, x_arg.stride(0), y.stride(0), N, eps, BLOCK_N
    )

    return y.reshape_as(x)
