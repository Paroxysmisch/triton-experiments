import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_cosine_similarity_avg_pool_kernel(
    x1_ptr, x2_ptr, output_ptr,
    batch, channels, height, width,
    kernel_size, stride, padding,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Calculate position in output
    out_idx = pid
    n = out_idx // (out_height * out_width)
    h = (out_idx // out_width) % out_height
    w = out_idx % out_width
    
    # Calculate input window start position
    h_start = h * stride - padding
    w_start = w * stride - padding
    
    # Initialize accumulators
    sum_prod = 0.0
    sum_x1_sq = 0.0
    sum_x2_sq = 0.0
    count = 0
    
    # Compute cosine similarity and average pool
    for kh in range(kernel_size):
        h_pos = h_start + kh
        if 0 <= h_pos < height:
            for kw in range(kernel_size):
                w_pos = w_start + kw
                if 0 <= w_pos < width:
                    # Accumulate dot products and squares
                    for c in range(channels):
                        idx = n * (channels * height * width) + c * (height * width) + h_pos * width + w_pos
                        x1_val = tl.load(x1_ptr + idx)
                        x2_val = tl.load(x2_ptr + idx)
                        sum_prod += x1_val * x2_val
                        sum_x1_sq += x1_val * x1_val
                        sum_x2_sq += x2_val * x2_val
                    count += 1
    
    # Compute final cosine similarity with average pooling
    denominator = tl.sqrt(sum_x1_sq * sum_x2_sq + eps)
    result = sum_prod / denominator if count > 0 else 0.0
    result = result / count if count > 0 else 0.0
    
    # Store result
    tl.store(output_ptr + out_idx, result)

def fused_avg_pool2d_cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes cosine similarity between x1 and x2 along channel dimension and applies 2D average pooling.
    
    Args:
        x1: Input tensor of shape (batch, channels, height, width)
        x2: Input tensor of shape (batch, channels, height, width)
        kernel_size: Size of the pooling kernel
        stride: Stride of the pooling operation (defaults to kernel_size if None)
        padding: Zero-padding added to both sides of input
        eps: Small value to prevent division by zero
    
    Returns:
        Tensor containing the fused cosine similarity and average pooling result
    """
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    assert x1.device.type == "cuda" and x2.device.type == "cuda", "Inputs must be CUDA tensors"
    
    if stride is None:
        stride = kernel_size
    
    batch, channels, height, width = x1.shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Allocate output tensor
    output = torch.empty((batch, out_height, out_width), device=x1.device, dtype=x1.dtype)
    
    # Launch kernel
    grid = (batch * out_height * out_width,)
    BLOCK_SIZE = 32
    
    fused_cosine_similarity_avg_pool_kernel[grid](
        x1_ptr=x1,
        x2_ptr=x2,
        output_ptr=output,
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
