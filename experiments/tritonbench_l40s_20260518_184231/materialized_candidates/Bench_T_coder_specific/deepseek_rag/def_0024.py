import triton
import triton.language as tl

@triton.jit
def least_squares_qr(A, b, *, mode='
