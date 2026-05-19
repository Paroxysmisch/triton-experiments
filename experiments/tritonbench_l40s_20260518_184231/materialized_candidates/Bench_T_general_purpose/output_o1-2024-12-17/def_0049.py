import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------------
# Triton kernel: Performs a single step of 2D convolution + Leaky ReLU on one output element.
# Each program handles exactly one (n, oc, oh, ow) element in the output.
# ------------------------------------------------------------------------------------
@triton.jit
def _leaky_relu_conv2d_kernel(
    IN_PTR,                     # input data pointer
    WT_PTR,                     # weight data pointer
    BIAS_PTR,                   # bias data pointer (can be None)
    OUT_PTR,                    # output data pointer
    N, C_IN, H_IN, W_IN,        # input tensor sizes
    C_OUT, K_H, K_W,            # weight/filter sizes
    stride_h, stride_w,         # stride
    pad_h, pad_w,               # padding
    dil_h, dil_w,               # dilation
    out_h, out_w,               # output height, width
    groups,                     # number of groups
