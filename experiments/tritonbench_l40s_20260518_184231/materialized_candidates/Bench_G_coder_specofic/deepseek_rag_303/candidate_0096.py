import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        # normalized
        x_hat = (x * rstd).to(x.dtype)
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)


def rmsnorm_forward(x, weight):
    # allocate output
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    
    # Triton kernel constraints
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    # Lauch Size Constraints
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # Heuristics for Triton num_warps, based on the feature size.
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    
    # Enqueue kernel
    _rms_norm_fwd_fused[(M,)](
        x_arg,
        y,
        weight,
        x_arg.stride(0),
        N,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y
