import math
import torch
import triton
import triton.language as tl

dtype = torch.float32

@triton.jit
def rbe_triton(x_p, stride_mb, stride_kn, out_p, stride_op,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, theta: tl.constexpr):
    # Prepare the pointers
    block_m = tl.program_id(0)
    block_k = tl.program_id(1)
    offset_m = block_m * BLOCK_SIZE_M
    offset_k = block_k * BLOCK_SIZE_K
    x_ptrs = x_p + tl.arange(0, BLOCK_SIZE_M) * stride_mb + tl.arange(0, BLOCK_SIZE_K) * stride_kn
    out_ptrs = out_p + tl.arange(0, BLOCK_SIZE_M) * stride_op + tl.arange(0, BLOCK_SIZE_K) * stride_kn
    x_re = x_ptrs + offset_k * 2
    out_re = out_ptrs + offset_k * 2
    # Load data into local variables
    x_re_local = tl.load(x_re)
    x_im_local = tl.load(x_re + offset_m + BLOCK_SIZE_K)
    # Apply position-dependent transformation
    f = get_freq_multi_tokens(theta)
    out_re_local = x_re_local * f
    out_im_local = x_im_local * f
    # Write back to the output tensor
    tl.store(out_re, out_re_local)
    tl.store(out_re + offset_m + BLOCK_SIZE_K, out_im_local)


def rbe_triton_wrapper(x, out, theta=10000):
    # Prepare parameters
    M, K = x.shape
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024
    grid = lambda:(M+BLOCK_SIZE_M-1)//BLOCK_SIZE_M, (K+BLOCK_SIZE_K-1)//BLOCK_SIZE_K
    # Launch the kernel
    rbe_triton[grid](x, M, K, out, M, BLOCK_SIZE_M, BLOCK_SIZE_K, theta)

# Define helper function to get frequency values
def get_freq_multi_tokens(theta, n_tokens=16):
    return torch.tensor([math.sin(theta/i) for i in range(1, n_tokens+1)] + 
                        [math.cos(theta/i) for i in range(1, n_tokens+1)])
