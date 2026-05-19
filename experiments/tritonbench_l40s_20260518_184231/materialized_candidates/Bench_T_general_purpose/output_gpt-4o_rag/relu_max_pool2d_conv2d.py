import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    A simple ReLU kernel implemented in Triton.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = tl.program_id(0) * BLOCK_SIZE + offsets
    mask = idx < N
    input_val = tl.load(input_ptr + idx, mask=mask)
    relu_val = tl.max(input_val, 0.0)
    tl.store(output_ptr + idx, relu_val, mask=mask)

def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    """
    Applies a 2D convolution over the input tensor, followed by max pooling and ReLU activation.
    """
    # Ensure input is on CUDA
    assert input.is_cuda, "Input tensor must be on a CUDA device for Triton ops."
    
    # Perform 2D convolution
    conv_output = F.conv2d(input, weight, bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
    
    # Perform max pooling
    pool_output = F.max_pool2d(conv_output, kernel_size=pool_kernel_size, stride=pool_stride or pool_kernel_size, padding=pool_padding, dilation=pool_dilation, ceil_mode=pool_ceil_mode)
    
    # Prepare output tensor for ReLU
    output = torch.empty_like(pool_output)
    N = pool_output.numel()
    BLOCK_SIZE = 256
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Apply ReLU using Triton
    relu_kernel[grid](pool_output, output, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

# Example usage
input_tensor = torch.randn(1, 3, 32, 32, device='cuda')
weight_tensor = torch.randn(6, 3, 5, 5, device='cuda')
output_tensor = relu_max_pool2d_conv2d(input_tensor, weight_tensor)
