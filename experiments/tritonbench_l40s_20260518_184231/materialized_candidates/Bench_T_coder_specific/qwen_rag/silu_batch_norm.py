import triton
import triton.language as tl

@triton.jit
def silu_batch_norm_kernel(
    X, Y, W, B, RMean, RVar, Out, 
    stride_x, stride_y, stride_w, stride_b, stride_rmean, stride_rvar, stride_out,
    N, C, H, W_,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data
    x = tl.load(X + offsets * stride_x, mask=offsets < N)
    w = tl.load(W + offsets * stride_w, mask=offsets < C)
    b = tl.load(B + offsets * stride_b, mask=offsets < C)
    rmean = tl.load(RMean + offsets * stride_rmean, mask=offsets < C)
    rvar = tl.load(RVar + offsets * stride_rvar, mask=offsets < C)

    # Batch normalization
    mean = rmean
    var = rvar + eps
    x_hat = (x - mean) / tl.sqrt(var)
    normalized = w * x_hat + b

    # SiLU activation
    sigma = 1 / (1 + tl.exp(-normalized))
    silu_output = normalized * sigma

    # Store result
    tl.store(Out + offsets * stride_out, silu_output, mask=offsets < N)
