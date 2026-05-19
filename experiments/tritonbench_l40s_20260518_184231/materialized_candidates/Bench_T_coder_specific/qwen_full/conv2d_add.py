import torch
import triton
import triton.language as tl
from triton.ops.conv2d import conv2d, conv2d_add
from triton.ops import conv2d as ref_conv2d

@triton.jit
def conv2d_add_triton(x, w, b, other, stride, padding, groups, alpha, dot_out):
    # Ensure x and w are 4D tensors
    assert x.dim() == 4 and w.dim() == 4
    # Get input and filter dimensions
    B, inC, iH, iW = x.shape
    outC, wC, kH, kW = w.shape
    # Calculate padding size
    padH = (kH - 1) * padding + 1
    padW = (kW - 1) * padding + 1
    # Ensure groups divide input and filter channels
    assert inC % groups == 0 and outC % groups == 0
    # Calculate output dimensions
    outH = (iH + 2 * padH - kH) // stride + 1
    outW = (iW + 2 * padW - kW) // stride + 1
    # Reshape input and weights for convolution
    x = x.view(B * groups, inC // groups, iH, iW)
    w = w.view(outC, inC // groups, kH, kW)
    # Prepare output tensor
    tl.static_assert((B * groups) % 32 == 0, "batch size must be divisible by 32")
    tl.static_assert((inC // groups) % 32 == 0, "in channel must be divisible by 32")
    tl.static_assert((outC // groups) % 32 == 0, "out channel must be divisible by 32")
    dot_out = dot_out.view(B * groups, outC // groups, outH, outW)
    # Call the conv2d_triton kernel
    conv2d_triton(x, w, b, stride, padding, groups, dot_out)
    # Apply bias and add other tensor
    dot_out += b
    dot_out = dot_out.to(x.dtype)
    dot_out += other * alpha

def conv2d_add_triton_wrapper(x, w, b=None, other=None, stride=1, padding=0, groups=1, alpha=1):
    # Ensure input dimensions are correct
    assert x.dim() == 4 and w.dim() == 4
    B, inC, iH, iW = x.shape
    outC, wC, kH, kW = w.shape
    # Prepare output tensor
    padH = (kH - 1) * padding + 1
    padW = (kW - 1) * padding + 1
    outH = (iH + 2 * padH - kH) // stride + 1
    outW = (iW + 2 * padW - kW) // stride + 1
    dot_out = torch.empty((B, outC, outH, outW), device=x.device, dtype=x.dtype)
    # Call the Triton kernel
    conv2d_add_triton(x, w, b, other, stride, padding, groups, alpha, dot_out)
    return dot_out
