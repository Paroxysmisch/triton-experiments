import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Process stride, padding, dilation
    stride = _pair(stride)
    dilation = _pair(dilation)
    
    # Handle padding
    if isinstance(padding, str):
        if padding not in ('valid', 'same'):
            raise ValueError("padding must be 'valid', 'same', int, or tuple")
    else:
        padding = _pair(padding)
    
    # Compute convolution
    output = F.conv2d(input, weight, None, stride, padding, dilation, groups)
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    # Add scaled other tensor if present
    if other is not None:
        output += alpha * other
    
    # Handle output tensor
    if out is not None:
        out.copy_(output)
        return out
    return output

import triton
import triton.language as tl

@triton.jit
def conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    alpha,
    B, C_in, H_in, W_in,
    C_out, K_H, K_W,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    H_out, W_out,
    HAS_BIAS: tl.constexpr,
    HAS_OTHER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    if pid_b >= B or pid_c >= C_out or pid_h >= H_out or pid_w >= W_out:
        return
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    group_size = C_in // groups
    c_start = (pid_c // (C_out // groups)) * group_size
    c_end = c_start + group_size
    
    for kh in range(K_H):
        h_in = pid_h * stride_h - padding_h + kh * dilation_h
        if h_in < 0 or h_in >= H_in:
            continue
        for kw in range(K_W):
            w_in = pid_w * stride_w - padding_w + kw * dilation_w
            if w_in < 0 or w_in >= W_in:
                continue
            for c in range(c_start, c_end):
                input_offset = pid_b * C_in * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in
                weight_offset = pid_c * (C_in // groups) * K_H * K_W + (c - c_start) * K_H * K_W + kh * K_W + kw
                input_val = tl.load(input_ptr + input_offset)
                weight_val = tl.load(weight_ptr + weight_offset)
                acc += input_val * weight_val
    
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + pid_c)
        acc += bias_val
    
    if HAS_OTHER:
        other_offset = pid_b * C_out * H_out * W_out + pid_c * H_out * W_out + pid_h * W_out + pid_w
        other_val = tl.load(other_ptr + other_offset)
        acc += alpha * other_val
    
    output_offset = pid_b * C_out * H_out * W_out + pid_c * H_out * W_out + pid_h * W_out + pid_w
    tl.store(output_ptr + output_offset, acc)
