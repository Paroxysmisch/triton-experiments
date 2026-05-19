import torch
import triton
import triton.language as tl
from typing import Union, Tuple
from torch import Tensor

@triton.jit
def sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    in_h, in_w,
    out_h, out_w,
    n_channels,
    batch_size,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate indices
    n = pid // (out_h * out_w * n_channels)
    remaining = pid % (out_h * out_w * n_channels)
    c = remaining // (out_h * out_w)
    h_out = (remaining // out_w) % out_h
    w_out = remaining % out_w
    
    # Calculate pooling window size
    h_stride = in_h // out_h
    w_stride = in_w // out_w
    h_start = h_out * h_stride
    w_start = w_out * w_stride
    
    # Compute average pooling
    sum_val = 0.0
    count = 0
    
    for h in range(h_start, min(h_start + h_stride, in_h)):
        for w in range(w_start, min(w_start + w_stride, in_w)):
            idx = (n * stride_n + c * stride_c + h * stride_h + w * stride_w)
            sum_val += tl.load(input_ptr + idx)
            count += 1
    
    # Calculate average
    avg = sum_val / count
    
    # Apply sigmoid activation: 1 / (1 + exp(-x))
    result = 1.0 / (1.0 + tl.exp(-avg))
    
    # Store result
    out_idx = (n * out_stride_n + c * out_stride_c + 
               h_out * out_stride_h + w_out * out_stride_w)
    tl.store(output_ptr + out_idx, result)

def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    # Input validation
    if not isinstance(input, Tensor):
        raise TypeError("Input must be a tensor")
    if input.dim() != 4:
        raise ValueError("Expected 4D tensor (batch_size, channels, height, width)")
        
    # Handle output_size parameter
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size
        
    # Get input dimensions
    batch_size, n_channels, in_h, in_w = input.shape
    
    # Create output tensor
    output = torch.empty((batch_size, n_channels, out_h, out_w), 
                        device=input.device, dtype=input.dtype)
    
    # Calculate strides
    stride_n = input.stride(0)
    stride_c = input.stride(1)
    stride_h = input.stride(2)
    stride_w = input.stride(3)
    
    out_stride_n = output.stride(0)
    out_stride_c = output.stride(1)
    out_stride_h = output.stride(2)
    out_stride_w = output.stride(3)
    
    # Launch kernel
    grid = (batch_size * n_channels * out_h * out_w,)
    sigmoid_adaptive_avg_pool2d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        in_h=in_h, in_w=in_w,
        out_h=out_h, out_w=out_w,
        n_channels=n_channels,
        batch_size=batch_size,
        stride_n=stride_n, stride_c=stride_c,
        stride_h=stride_h, stride_w=stride_w,
        out_stride_n=out_stride_n,
        out_stride_c=out_stride_c,
        out_stride_h=out_stride_h,
        out_stride_w=out_stride_w,
        BLOCK_SIZE=32
    )
    
    return output
