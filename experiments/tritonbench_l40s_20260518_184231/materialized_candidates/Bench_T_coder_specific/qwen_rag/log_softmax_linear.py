import triton
import triton.language as tl

@triton.jit
def linear_transform_kernel(
    X, W, Y,
    stride_X, stride_W, stride_Y,
    M, K, N
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    acc = tl.zeros((N,), dtype=tl.float32)
    for k in range(K):
        acc += X[row * stride_X + k] * W[k * stride_W + col]
    
    Y[row * stride_Y + col] = acc
