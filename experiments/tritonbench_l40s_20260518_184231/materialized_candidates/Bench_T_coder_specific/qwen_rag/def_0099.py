import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh

@triton.jit
def gelu_exact_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    X_ptr = X + pid * BLOCK_SIZE
    Y_ptr = Y + pid * BLOCK_SIZE
    
    for i in range(BLOCK_SIZE):
        x = X_ptr[i]
        x_fp32 = x.to(tl.float32)
        x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
        Y_ptr[i] = x_gelu

@triton.jit
def gelu_tanh_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    X_ptr = X + pid * BLOCK_SIZE
    Y_ptr = Y + pid * BLOCK_SIZE
    
    for i in range(BLOCK_SIZE):
        x = X_ptr[i]
        x_fp32 = x.to(tl.float32)
        x_gelu = (
            0.5
            * x_fp32
            * (
                1
                + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))
            )
        )
        Y_ptr[i] = x_gelu
