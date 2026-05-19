import triton
import triton.language as tl
import torch

@triton.jit
def diag_ssm_forward_kernel(
    x_ptr, s_ptr, lambda_ptr, y_ptr,
    stride_xb, stride_xd, stride_xt,
    stride_sb, stride_sd,
    stride_lb,
    stride_yb, stride_yd, stride_yt,
    batch_size, dim, length,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch and dim indices
    batch_idx = pid // (dim // BLOCK_SIZE)
    dim_idx = (pid % (dim // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Load block of states
    s_offs = batch_idx * stride_sb + dim_idx * stride_sd
    s = tl.load(s_ptr + s_offs + tl.arange(0, BLOCK_SIZE))
    
    # Load block of Lambda values
    l_offs = batch_idx * stride_lb + dim_idx
    Lambda = tl.load(lambda_ptr + l_offs + tl.arange(0, BLOCK_SIZE))
    
    # Iterate through sequence
    for t in range(length):
        # Load input x
        x_offs = batch_idx * stride_xb + dim_idx * stride_xd + t * stride_xt
        x = tl.load(x_ptr + x_offs + tl.arange(0, BLOCK_SIZE))
        
        # Update state: s = s * Lambda + x
        s = s * Lambda + x
        
        # Store output y
        y_offs = batch_idx * stride_yb + dim_idx * stride_yd + t * stride_yt
        tl.store(y_ptr + y_offs + tl.arange(0, BLOCK_SIZE), s)

# PyTorch wrapper function
class SSMForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, Lambda):
        batch_size, dim, length = x.shape
        device = x.device
        
        # Save for backward
        ctx.save_for_backward(x, s, Lambda)
        ctx.dims = (batch_size, dim, length)
        
        # Allocate output
        y = torch.empty_like(x)
        
        # Configure grid
        BLOCK_SIZE = 32
        grid = (batch_size * (dim // BLOCK_SIZE),)
        
        # Launch kernel
        diag_ssm_forward_kernel[grid](
            x, s, Lambda, y,
            x.stride(0), x.stride(1), x.stride(2),
            s.stride(0), s.stride(1),
            Lambda.stride(0),
            y.stride(0), y.stride(1), y.stride(2),
            batch_size, dim, length,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # Backward pass implementation would go here
        pass

# User-facing function
def ssm_forward(x, s, Lambda):
    return SSMForward.apply(x, s, Lambda)
