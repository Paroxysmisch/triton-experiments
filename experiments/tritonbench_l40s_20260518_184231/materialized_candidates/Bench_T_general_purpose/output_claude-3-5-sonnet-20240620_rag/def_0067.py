import torch
import triton
import triton.language as tl
import math

@triton.jit
def _adaptive_avg_pool2d_kernel(
    x_ptr, output_ptr,
    in_h, in_w, out_h, out_w,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate position
    n = pid // (out_h * out_w)
    hw = pid % (out_h * out_w)
    oh = hw // out_w
    ow = hw % out_w
    
    # Calculate input window size
    stride_h = in_h // out_h
    stride_w = in_w // out_w
    
    # Calculate start positions
    start_h = oh * stride_h
    start_w = ow * stride_w
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = 0
    
    # Perform pooling
    for ih in range(start_h, min(start_h + stride_h, in_h)):
        for iw in range(start_w, min(start_w + stride_w, in_w)):
            idx = n * stride_n + ih * stride_h + iw * stride_w
            acc += tl.load(x_ptr + idx)
            count += 1
    
    # Calculate average
    acc = acc / float(count)
    
    # Store result
    out_idx = n * out_stride_n + oh * out_stride_h + ow * out_stride_w
    tl.store(output_ptr + out_idx, acc)

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr, x2_ptr, output_ptr,
    n1, n2, dim,
    p, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate indices
    i = pid // n2
    j = pid % n2
    
    if i >= n1:
        return
        
    # Initialize accumulator
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Compute distance
    for k in range(0, dim):
        x1_val = tl.load(x1_ptr + i * dim + k)
        x2_val = tl.load(x2_ptr + j * dim + k)
        diff = tl.abs(x1_val - x2_val)
        if p == 2.0:
            acc += diff * diff
        else:
            acc += tl.power(diff, p)
    
    # Apply final operations
    if p == 2.0:
        result = tl.sqrt(acc + eps)
    else:
        result = tl.power(acc + eps, 1.0/p)
        
    # Store result
    tl.store(output_ptr + i * n2 + j, result)

class FusedPairwiseDistanceAdaptiveAvgPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, output_size, p=2.0, eps=1e-6):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
            
        # Get input dimensions
        batch1, channels1, height1, width1 = x1.shape
        batch2, channels2, height2, width2 = x2.shape
        
        # Create output tensors for pooling
        pooled1 = torch.empty((batch1, channels1, output_size[0], output_size[1]), 
                            device=x1.device, dtype=x1.dtype)
        pooled2 = torch.empty((batch2, channels2, output_size[0], output_size[1]), 
                            device=x2.device, dtype=x2.dtype)
        
        # Launch pooling kernels
        grid1 = (batch1 * output_size[0] * output_size[1],)
        _adaptive_avg_pool2d_kernel[grid1](
            x1, pooled1,
            height1, width1, output_size[0], output_size[1],
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            pooled1.stride(0), pooled1.stride(1), pooled1.stride(2), pooled1.stride(3),
            BLOCK_SIZE=32
        )
        
        grid2 = (batch2 * output_size[0] * output_size[1],)
        _adaptive_avg_pool2d_kernel[grid2](
            x2, pooled2,
            height2, width2, output_size[0], output_size[1],
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            pooled2.stride(0), pooled2.stride(1), pooled2.stride(2), pooled2.stride(3),
            BLOCK_SIZE=32
        )
        
        # Reshape pooled outputs for distance calculation
        pooled1_flat = pooled1.reshape(batch1, -1)
        pooled2_flat = pooled2.reshape(batch2, -1)
        
        # Create output tensor for distances
        distances = torch.empty((batch1, batch2), device=x1.device, dtype=x1.dtype)
        
        # Launch distance kernel
        grid = (batch1 * batch2,)
        _pairwise_distance_kernel[grid](
            pooled1_flat, pooled2_flat, distances,
            batch1, batch2, pooled1_flat.size(1),
            p, eps,
            BLOCK_SIZE=32
        )
        
        ctx.save_for_backward(pooled1_flat, pooled2_flat, distances)
        ctx.p = p
        ctx.eps = eps
        
        return distances

def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_size: int or tuple,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Fused operation for adaptive average pooling and pairwise distance calculation.
    
    Args:
        x1 (Tensor): First input tensor
        x2 (Tensor): Second input tensor
        output_size (int or tuple): Target output size for adaptive pooling
        p (float): The norm degree for distance calculation
        eps (float): Small value to avoid division by zero
        keepdim (bool): Whether to keep the reduced dimension
        
    Returns:
        Tensor: Pairwise distances between pooled tensors
    """
    result = FusedPairwiseDistanceAdaptiveAvgPool2d.apply(x1, x2, output_size, p, eps)
    if keepdim:
        result = result.unsqueeze(-1)
    return result
