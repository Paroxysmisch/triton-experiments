import triton
import triton.language as tl

# Block sizes and epsilon for numerical stability
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
EPS = 1e-5

@triton.jit
def ff_llama_kernel(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_w1k,
