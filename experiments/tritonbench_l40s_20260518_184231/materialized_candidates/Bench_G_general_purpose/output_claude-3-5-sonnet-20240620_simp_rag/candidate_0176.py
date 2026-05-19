import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor 
    stride_x_row,  # stride between rows
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_N: tl.constexpr,  # size of block to process
):
    # Get the row index this thread block processes
    row = tl.program_id(0)
    
    # Offset input/output pointers to current row
    X += row * stride_x_row  
    Y += row * stride_x_row

    # Create block of column indices
    cols = tl.arange(0, BLOCK_N)
    
    # Load input values with masking for out-of-bounds
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    
    # Calculate L2 norm
    var = tl.sum(x * x, axis=0) 
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize and store result
    y = x * rstd
    tl.store(Y + cols, y, mask=cols < N)

def _l2_norm_fwd(x, eps=1e-6):
    # Reshape input to 2D
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    
    # Ensure input is contiguous
    if x.stride(-1) != 1:
        x = x.contiguous()
    
    # Get dimensions
    M, N = x.shape
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Calculate block size
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    
    # Check size constraints
    if N > BLOCK_N:
        raise RuntimeError("Feature dim must be < 64KB")
        
    # Launch kernel
    with torch.cuda.device(x.device.index):
        _l2_norm_fwd_1pass_kernel[(M,)](
            x, y, x.stride(0), N, eps, BLOCK_N,
        )
    
    # Reshape output back to original shape
    return y.reshape(x_shape_og)
