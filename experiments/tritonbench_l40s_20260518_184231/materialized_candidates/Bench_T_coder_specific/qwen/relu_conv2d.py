import torch
import triton
import triton.language as tl

# Define constants
BLOCK_SIZE_I = 32
BLOCK_SIZE_K = 8

@torch.jit.script
def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Get tensor shapes
    N, C, H, W = input.shape
    OC, IC, KH, KW = weight.shape
    
    # Pad the input tensor
    padded_input = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)
    
    # Create output tensors
    output_shape = (N, OC, H, W)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    
    # Run convolution kernel
    conv2d_kernel[output.numel()](padded_input, weight, output, bias, stride, padding, dilation, groups, N, C, H, W, OC, KH, KW, BLOCK_SIZE_I, BLOCK_SIZE_K)
    
    # Run ReLU kernel
    relu_kernel[output.numel()](output, output, N, H, W, OC, BLOCK_SIZE_I, BLOCK_SIZE_K)
    
    return output
