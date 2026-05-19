import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, out_channels, in_h, in_w, out_h, out_w,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, approximate,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(out_channels, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_h * out_w, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    out_channel_offset = group_id * (out_channels // groups)
    in_channel_per_group = in_channels // groups
    
    # Load input, weight, and compute convolution
    input_block_ptr = input_ptr + group_id * in_channel_per_group * in_h * in_w
    weight_block_ptr = weight_ptr + out_channel_offset * in_channel_per_group * kernel_h * kernel_w
    
    # Compute convolution
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ic in range(0, in_channel_per_group, BLOCK_SIZE_K):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                ih = tl.arange(0, BLOCK_SIZE_N) // out_w * stride_h - padding_h + kh * dilation_h
                iw = tl.arange(0, BLOCK_SIZE_N) % out_w * stride_w - padding_w + kw * dilation_w
                input_mask = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
                weight_mask = (tl.arange(0, BLOCK_SIZE_M) < out_channels) & (ic + tl.arange(0, BLOCK_SIZE_K) < in_channel_per_group)
                
                input_block = tl.load(input_block_ptr + ih * in_w + iw + ic * in_h * in_w, mask=input_mask, other=0.0)
                weight_block = tl.load(weight_block_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * in_channel_per_group * kernel_h * kernel_w + 
                                       (ic + tl.arange(0, BLOCK_SIZE_K)[None, :]) * kernel_h * kernel_w + 
                                       kh * kernel_w + kw, mask=weight_mask, other=0.0)
                
                acc += tl.dot(weight_block, input_block)
    
    # Apply bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_channel_offset + tl.arange(0, BLOCK_SIZE_M))
        acc += bias[:, None]
    
    # Apply GELU activation
    if approximate == 'none':
        # Exact GELU
        acc = 0.5 * acc * (1 + tl.erf(acc / math.sqrt(2)))
    elif approximate == 'tanh':
        # Tanh approximation
        acc = 0.5 * acc * (1 + tl.tanh(math.sqrt(2 / math.pi) * (acc + 0.044715 * acc**3)))
    
    # Store output
    output_offset = group_id * (out_channels // groups) * out_h * out_w
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * out_h * out_w + 
             tl.arange(0, BLOCK_SIZE_N)[None, :], acc)

def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                groups: int = 1, approximate: str = 'none', out: torch.Tensor = None) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda, "Input and weight must be CUDA tensors"
    
    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((batch, out_channels, out_h, out_w), device=input.device, dtype=input.dtype)
    
    # Prepare bias
    if bias is not None:
        assert bias.is_cuda, "Bias must be a CUDA tensor"
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = None
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Launch kernel
    grid = lambda meta: (
        triton.cdiv(out_channels, BLOCK_SIZE_M) * triton.cdiv(out_h * out_w, BLOCK_SIZE_N) * groups,
    )
    gelu_conv2d_kernel[grid](
        input.data_ptr(), weight.data_ptr(), bias_ptr, out.data_ptr(),
        batch, in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        groups, approximate,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out
