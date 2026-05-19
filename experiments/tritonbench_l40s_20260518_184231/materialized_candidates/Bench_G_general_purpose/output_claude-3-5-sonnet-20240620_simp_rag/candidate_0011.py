import torch
import triton
import triton.language as tl
import math

@triton.jit
def rms_matmul_rbe_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_length, num_heads, head_dim,
    # Strides
    stride_xb, stride_xs, stride_xh,
    stride_wb, stride_wh,
    stride_ob, stride_os, stride_oh,
    # RMS norm parameters
    eps: tl.float32,
    # RBE parameters
    theta_base: tl.float32,
    BLOCK_SIZE: tl.constexpr):
    
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and sequence indices
    batch_id = pid // (seq_length * num_heads)
    seq_id = (pid // num_heads) % seq_length
    head_id = pid % num_heads
    
    # Compute input/output pointers
    x_start_ptr = x_ptr + batch_id * stride_xb + seq_id * stride_xs + head_id * stride_xh
    w_start_ptr = weight_ptr + head_id * stride_wh
    out_start_ptr = output_ptr + batch_id * stride_ob + seq_id * stride_os + head_id * stride_oh
    
    # Load offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    
    # Load input
    x_row = tl.load(x_start_ptr + offsets, mask=mask, other=0.0)
    
    # RMS Normalization
    x_squared = x_row * x_row
    rms = tl.sqrt(tl.sum(x_squared) / head_dim + eps)
    x_normalized = x_row / rms
    
    # Apply RBE
    position = tl.float32(seq_id)
    freq = tl.exp(-theta_base * tl.log(10000.0) * tl.arange(0, head_dim // 2) / (head_dim // 2))
    theta = position * freq
    
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    
    x_even = x_normalized[::2]
    x_odd = x_normalized[1::2]
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_odd * cos + x_even * sin
    
    # Interleave even and odd elements
    x_rbe = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x_rbe[::2] = x_rotated_even
    x_rbe[1::2] = x_rotated_odd
    
    # Matrix multiplication
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, head_dim, BLOCK_SIZE):
        weight = tl.load(w_start_ptr + i + offsets, mask=mask, other=0.0)
        acc += x_rbe * weight
    
    # Store output
    tl.store(out_start_ptr + offsets, acc, mask=mask)

class RMSMatMulRBE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6, theta_base=10000.0):
        # Save inputs for backward pass
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.theta_base = theta_base
        
        # Extract dimensions
        batch_size, seq_length, num_heads, head_dim = x.shape
        
        # Allocate output
        output = torch.empty_like(x)
        
        # Compute grid and block sizes
        grid = (batch_size * seq_length * num_heads,)
        BLOCK_SIZE = triton.next_power_of_2(head_dim)
        
        # Launch kernel
        rms_matmul_rbe_kernel[grid](
            x, weight, output,
            batch_size, seq_length, num_heads, head_dim,
            x.stride(0), x.stride(1), x.stride(2),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            eps, theta_base,
            num_warps=4,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

def rms_matmul_rbe_wrapper(x: torch.Tensor, 
                          weight: torch.Tensor,
                          eps: float = 1e-6,
                          theta_base: float = 10000.0) -> torch.Tensor:
    """
    Wrapper function for RMS matrix multiplication with rotary embeddings.
    
    Args:
        x: Input tensor of shape [batch_size, seq_length, num_heads, head_dim]
        weight: Weight matrix of shape [num_heads, head_dim]
        eps: Epsilon for RMS normalization
        theta_base: Base for rotary embeddings
    
    Returns:
        Output tensor of shape [batch_size, seq_length, num_heads, head_dim]
    """
    assert x.is_cuda and weight.is_cuda, "Input tensors must be on GPU"
    assert x.is_contiguous() and weight.is_contiguous(), "Input tensors must be contiguous"
    
    return RMSMatMulRBE.apply(x, weight, eps, theta_base)
