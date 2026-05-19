import triton
import triton.language as tl
import torch

@triton.jit
def _relu_batch_norm_conv2d_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr,
    bias_ptr, bn_weight_ptr, bn_bias_ptr,
    running_mean_ptr, running_var_ptr,
    # Matrix dimensions
    batch_size, in_channels, out_channels,
    height, width, kernel_h, kernel_w,
    # Parameters
    stride, padding, dilation, groups,
    training, momentum, eps,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Calculate position in output
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(out_channels, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(batch_size * height * width, BLOCK_SIZE_N)
    
    # Position in output matrix
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Load block pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Convolution
    for k in range(0, in_channels // groups):
        # Load weights and input
        w = tl.load(weight_ptr + offs_m * kernel_h * kernel_w + k)
        x = tl.load(input_ptr + offs_n * in_channels + k)
        
        # Compute convolution
        acc += tl.dot(w, x)
    
    # Load batch norm parameters
    if running_mean_ptr is not None:
        mean = tl.load(running_mean_ptr + offs_m)
        var = tl.load(running_var_ptr + offs_m)
        gamma = tl.load(bn_weight_ptr + offs_m) if bn_weight_ptr is not None else 1.0
        beta = tl.load(bn_bias_ptr + offs_m) if bn_bias_ptr is not None else 0.0
        
        # Apply batch normalization
        acc = (acc - mean[:, None]) / tl.sqrt(var[:, None] + eps)
        acc = acc * gamma[:, None] + beta[:, None]
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store output
    output = acc.to(tl.float16)
    tl.store(output_ptr + offs_m * height * width + offs_n, output)

def relu_batch_norm_conv2d(
    input, weight, bias=None,
    stride=1, padding=0, dilation=1, groups=1,
    running_mean=None, running_var=None,
    bn_weight=None, bn_bias=None,
    training=False, momentum=0.1, eps=1e-5,
    inplace=False
):
    # Input validation and shape calculations
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Allocate output tensor
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        device=input.device, dtype=input.dtype
    )
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch kernel
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M) *
        triton.cdiv(batch_size * out_height * out_width, BLOCK_SIZE_N),
    )
    
    _relu_batch_norm_conv2d_kernel[grid](
        input.data_ptr(), weight.data_ptr(), output.data_ptr(),
        bias.data_ptr() if bias is not None else None,
        bn_weight.data_ptr() if bn_weight is not None else None,
        bn_bias.data_ptr() if bn_bias is not None else None,
        running_mean.data_ptr() if running_mean is not None else None,
        running_var.data_ptr() if running_var is not None else None,
        batch_size, in_channels, out_channels,
        height, width, kernel_h, kernel_w,
        stride, padding, dilation, groups,
        training, momentum, eps,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output
