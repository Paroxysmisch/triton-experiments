import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_accum_kernel(
    x_ptr, out_ptr,
    NC, H_in, W_in, OH, OW,
    stride_nc, stride_h, stride_w,
