import triton
import triton.language as tl
import torch

@triton.jit
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': w})
        for w in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
def logsumexp_fwd_kernel(
    X,  # Pointer to input tensor
    Z,  # Pointer to output tensor
    stride_xn,  # Stride for input tensor along N dimension
    stride_xd,  # Stride for input tensor along D dimension
    stride_z,   # Stride for output tensor
    N,  # Batch size
    D,  # Feature dimension size
    HAS_SCALE,  # Whether to use scaling
    BLOCK_SIZE: tl.constexpr,  # Size of block for parallel processing
):
    # Program ID for parallel execution
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)
    
    # Calculate offsets and ranges
    o_d = i_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m_d = o_d < D
    
    # Load input block
    x_ptr = X + i_n * stride_xn + o_d * stride_xd
    b_x = tl.load(x_ptr, mask=m_d, other=-float('inf'))
    
    # Apply scaling if needed
    if HAS_SCALE:
        b_x = b_x / D ** 0.5
    
    # Compute maximum for numerical stability
    b_m = tl.max(b_x, axis=0)
    
    # Compute exp(x - max) and sum
    b_z = tl.exp(b_x - b_m)
    b_z = tl.sum(b_z, axis=0)
    
    # Compute log and add back the max
    b_z = tl.log(b_z) + b_m
    
    # Store result
    z_ptr = Z + i_n
    tl.store(z_ptr, b_z)

class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=False):
        # Save input for backward pass
        ctx.save_for_backward(x)
        ctx.scale = scale
        
        # Get dimensions
        N = x.shape[0]
        D = x.shape[-1]
        B = 128  # Default block size
        
        # Reshape input if needed
        x_view = x.view(-1, D)
        N_view = x_view.shape[0]
        ND = triton.cdiv(D, B)
        
        # Create output tensor
        z = torch.empty(N_view, dtype=x.dtype, device=x.device)
        
        # Launch kernel
        logsumexp_fwd_kernel[(N_view, ND)](
            x_view, z,
            x_view.stride(0), x_view.stride(1),
            z.stride(0),
            N_view, D,
            scale,
            BLOCK_SIZE=B
        )
        
        # Reshape output if needed
        z = z.view(N)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = ctx.scale
        
        # Implement backward pass if needed
        # This is left as an exercise
        return grad_output, None

def logsumexp(x, scale=False):
    return LogSumExp.apply(x, scale)
