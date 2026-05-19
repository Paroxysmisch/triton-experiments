import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(X_ptr, Y_ptr, W_ptr, stride, N, BLOCK_SIZE: tl.constexpr):
    # Define shared memory for partial sums and counts
    x = tl.load(X_ptr + tl.program_id(0) * stride + tl.arange(0, BLOCK_SIZE))
    x2 = x * x
    s = tl.sum(x2, axis=0)
    c = tl.sum(x, axis=0)

    # Compute mean and variance
    mean = c / N
    var = s / N - mean * mean

    # Compute inverse standard deviation (rstd)
    rstd = tl.rsqrt(var + 1e-5)

    # Compute normalized output
    y = x * rstd * W_ptr[tl.arange(0, BLOCK_SIZE)]

    # Store the result in output tensor
    tl.store(Y_ptr + tl.program_id(0) * stride + tl.arange(0, BLOCK_SIZE), y)
