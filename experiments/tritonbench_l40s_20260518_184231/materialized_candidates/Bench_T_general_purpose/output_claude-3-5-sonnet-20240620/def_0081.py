import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Union, Tuple

@triton.jit
def sigmoid_adaptive_pool_kernel(
    # Pointers to input/output tensors
    input_ptr, output_ptr,
    # Input dimensions
    in_h, in_w, 
    # Output dimensions
    out_h, out_w,
    # Strides for input/output
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    # Other parameters
    batch_size, channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    
    # Calculate indices
    n = pid // (channels * out_h * out_w)
    c = (pid // (out_h * out_w)) % channels
    oh = (pid // out_w) % out_h
    ow = pid % out_w
    
    # Calculate pooling region size
    h_start = (oh * in_h) // out_h
    h_end = ((oh + 1) * in_h) // out_h
    w_start = (ow * in_w) // out_w
    w_end = ((ow + 1) * in_w) // out_w
    
    # Initialize accumulator
    acc = 0.0
    count = 0
    
    # Compute average pooling
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            inp_idx = (n * stride_in_n + c * stride_in_c + 
                      h * stride_in_h + w * stride_in_w)
            acc += tl.load(input_ptr + inp_idx)
            count += 1
    
    # Calculate average
    if count > 0:
        acc = acc / float(count)
    
    # Apply sigmoid: 1 / (1 + exp(-x))
    acc = 1.0 / (1.0 + tl.exp(-acc))
    
    # Store result
    out_idx = (n * stride_out_n + c * stride_out_c + 
               oh * stride_out_h + ow * stride_out_w)
    tl.store(output_ptr + out_idx, acc)

def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    # Input validation
    if not isinstance(input, Tensor):
        raise TypeError("Input must be a tensor")
    
    if input.dim() != 4:
        raise ValueError("Expected 4D tensor (N, C, H, W)")
    
    # Handle output_size parameter
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size
    
    # Get input dimensions
    batch_size, channels, in_h, in_w = input.shape
    
    # Create output tensor
    output = torch.empty(batch_size, channels, out_h, out_w, 
                        device=input.device, dtype=input.dtype)
    
    # Get strides
    stride_in_n, stride_in_c, stride_in_h, stride_in_w = input.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = output.stride()
    
    # Launch kernel
    grid = (batch_size * channels * out_h * out_w,)
    sigmoid_adaptive_pool_kernel[grid](
        input_ptr=input, 
        output_ptr=output,
        in_h=in_h, in_w=in_w,
        out_h=out_h, out_w=out_w,
        stride_in_n=stride_in_n, stride_in_c=stride_in_c,
        stride_in_h=stride_in_h, stride_in_w=stride_in_w,
        stride_out_n=stride_out_n, stride_out_c=stride_out_c,
        stride_out_h=stride_out_h, stride_out_w=stride_out_w,
        batch_size=batch_size, channels=channels,
        BLOCK_SIZE=32,
    )
    
    return output
