import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(X, W, Y, pid, mask, N, eps, stride_x, stride_y, stride_w):
    # Each block handles one row of X
    row = pid
    # Load the current row of X
    x = tl.load(X + row * stride_x, mask=mask)
    # Compute the variance
    var = tl.sum(x * x, axis=0) / N
    # Compute the reciprocal of the square root of variance plus epsilon
    rrms = 1.0 / tl.sqrt(var + eps)
    # Normalize and scale by weights
    y = (x * rrms).to(Y.dtype.element_ty) * W
    # Store the result in Y
    tl.store(Y + row * stride_y, y, mask=mask)
