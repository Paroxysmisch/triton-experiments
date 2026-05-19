import triton
import triton.language as tl

@triton.jit
def relu_kernel(X, Y, stride_x, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x = tl.load(X + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(Y + offsets, y, mask=mask)

@triton.jit
def fractional_max_pool2d_kernel(X, Y, indices, stride_x, stride_y, kernel_h, kernel_w, output_h, output_w, input_h, input_w, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    
    for i in range(output_h):
        for j in range(output_w):
            start_h = i * stride
            start_w = j * stride
            end_h = min(start_h + kernel_h, input_h)
            end_w = min(start_w + kernel_w, input_w)
            
            max_val = -float('inf')
            max_idx = -1
            
            for h in range(start_h, end_h):
                for w in range(start_w, end_w):
                    idx = h * input_w + w
                    x = tl.load(X + idx, mask=mask)
                    if x > max_val:
                        max_val = x
                        max_idx = idx
            
            output_idx = (i * output_w + j) * BLOCK_SIZE + offsets
            tl.store(Y + output_idx, max_val, mask=mask)
            if indices is not None:
                tl.store(indices + output_idx, max_idx, mask=mask)

import torch
import triton
import triton.language as tl

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Ensure input is a 4D tensor (N, C, H, W)
    assert input.dim() == 4, "Input tensor must be 4D (N, C, H, W)"
    
    # Determine output size
    if output_size is not None and output_ratio is not None:
        raise ValueError("Only one of output_size or output_ratio can be specified")
    elif output_size is not None:
        output_h, output_w = output_size
    elif output_ratio is not None:
        input_h, input_w = input.shape[2], input.shape[3]
        output_h = int(input_h * output_ratio[0])
        output_w = int(input_w * output_ratio[1])
    else:
        raise ValueError("Either output_size or output_ratio must be specified")
    
    # Apply ReLU
    input_relu = torch.relu(input)
    
    # Determine kernel size
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    # Determine stride
    stride_h = (input.shape[2] - kernel_h) // (output_h - 1) + 1
    stride_w = (input.shape[3] - kernel_w) // (output_w - 1) + 1
    
    # Initialize output and indices tensors
    output = torch.empty((input.shape[0], input.shape[1], output_h, output_w), device=input.device, dtype=input.dtype)
    indices = None
    if return_indices:
        indices = torch.empty((input.shape[0], input.shape[1], output_h, output_w), device=input.device, dtype=torch.int64)
    
    # Launch Triton kernels
    grid = lambda meta: (input.numel() // meta['BLOCK_SIZE'],)
    relu_kernel[grid](input_relu, input_relu, input.stride(0), BLOCK_SIZE=1024)
    fractional_max_pool2d_kernel[grid](input_relu, output, indices, input.stride(0), input.stride(1), kernel_h, kernel_w, output_h, output_w, input.shape[2], input.shape[3], stride_h, BLOCK_SIZE=1024)
    
    if return_indices:
        return output, indices
    else:
        return output
