import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr,
):
    # Get the row index we're processing
    row = tl.program_id(0)
    
    # Offset pointers to the current row
    X += row * stride_x_row
    Y += row * stride_x_row
    
    # Create a range for accessing elements in the block
    cols = tl.arange(0, BLOCK_N)
    
    # Load input values with masking for out-of-bounds access
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    
    # Compute variance (sum of squares)
    var = tl.sum(xbar * xbar, axis=0)
    
    # Compute reciprocal of standard deviation
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize and store the result
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)

def _l2_norm_fwd(x, eps=1e-6):
    # Save original shape and reshape to 2D
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    
    # Ensure input is contiguous
    if x.stride(-1) != 1:
        x = x.contiguous()
    
    M, N = x.shape
    assert x.stride(-1) == 1
    
    # Create output tensor
    y = torch.empty_like(x)
    assert y.stride(-1) == 1
    
    # Calculate block size (must be power of 2 and fit in shared memory)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    
    if N > BLOCK_N:
        raise RuntimeError("Feature dim must be < 64KB")
        
    # Launch kernel
    with torch.cuda.device(x.device.index):
        _l2_norm_fwd_1pass_kernel[(M,)](
            x, y, x.stride(0), N, eps, BLOCK_N,
        )
    
    return y.reshape(x_shape_og)
