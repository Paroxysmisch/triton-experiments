@triton.jit
def log_softmax_kernel(Y, X, maxes, sum_weights, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...

@triton.jit
def log_softmax_backward_kernel(dY, dX, Y, sum_weights, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...
