import triton
import triton.language as tl
import torch

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X_ptr,  # pointer to input tensor
    Y_ptr,  # pointer to output tensor
    stride_x_row,  # stride between rows
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_N: tl.constexpr,  # number of elements per block
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Compute pointer offset for this row
    row_start_ptr = X_ptr + row_idx * stride_x_row
    
    # Create offsets for this block
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    
    # Load input values
    x = tl.load(row_start_ptr + offs, mask=mask, other=0.0)
    
    # Compute sum of squares for variance
    x_sq = x * x
    var = tl.sum(x_sq, axis=0)
    
    # Compute reciprocal of square root of variance
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize the values
    y = x * rstd
    
    # Store the results
    row_out_ptr = Y_ptr + row_idx * N
    tl.store(row_out_ptr + offs, y, mask=mask)

def _l2_norm_fwd(x, eps=1e-12):
    # Handle input reshaping
    orig_shape = x.shape
    if x.dim() != 2:
        x = x.reshape(x.shape[0], -1)
    
    # Make input contiguous if needed
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Get dimensions
    M, N = x.shape
    
    # Calculate BLOCK_N based on element size (assuming float32)
    # Maximum shared memory size is 64KB
    BLOCK_N = min(N, (64 * 1024) // x.element_size())
    
    # Check if N is too large
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} exceeds maximum block size {BLOCK_N}")
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Launch kernel
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        x.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )
    
    # Reshape output to original dimensions
    return y.reshape(orig_shape)
