import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, 
                  stride_h, stride_w, padding_h, padding_w, 
                  N, C_in, H, W, C_out, kH, kW, group, 
                  BLOCK_SIZE: tl.constexpr):
    # Define block and grid sizes
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Calculate output indices
    h_out = row * stride_h
    w_out = col * stride_w
    
    # Initialize output
    output_val = 0.0
    
    # Iterate over the kernel
    for kh in range(kH):
        for kw in range(kW):
            h_in = h_out + kh - padding_h
            w_in = w_out + kw - padding_w
            
            if 0 <= h_in < H and 0 <= w_in < W:
                input_val = tl.load(input_ptr + h_in * W + w_in)
                weight_val = tl.load(weight_ptr + kh * kW + kw)
                output_val += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        output_val += tl.load(bias_ptr)
    
    # Store the result
    tl.store(output_ptr + row * W + col, output_val)

import torch
import torch.nn.functional as F

def dropout_relu_batch_norm_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, 
                                    stride=1, padding=0, dilation=1, groups=1, 
                                    p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Get input dimensions
    N, C_in, H, W = input.shape
    C_out = weight.shape[0]
    kH, kW = weight.shape[2], weight.shape[3]
    
    # Calculate padding
    padding_h = padding
    padding_w = padding
    
    # Allocate output tensor
    output = torch.empty((N, C_out, (H + 2 * padding_h - kH) // stride + 1, 
                            (W + 2 * padding_w - kW) // stride + 1), device=input.device)
    
    # Launch the Triton kernel
    grid = (output.shape[2], output.shape[3])
    conv2d_kernel[grid](input, weight, bias, output, 
                        stride, stride, padding_h, padding_w, 
                        N, C_in, H, W, C_out, kH, kW, groups)
    
    # Apply batch normalization
    output = F.batch_norm(output, running_mean=None, running_var=None, training=training)
    
    # Apply ReLU activation
    output = F.relu(output, inplace=inplace)
    
    # Apply dropout
    if training:
        output = F.dropout(output, p=p, training=training)
    
    return output
