import triton
import triton.language as tl
import torch

@triton.jit
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=5),
    ],
    key=['ncols']
)
def _swiglu_fwd_kernel(
    OUT, X, Y,  # Pointers to tensors
    stride_xm, stride_xn,  # Strides for X tensor
    stride_ym, stride_yn,  # Strides for Y tensor
    stride_om, stride_on,  # Strides for output tensor
    nrows, ncols,  # Shape of the matrices
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Row and column block indices
    row_idx = pid // triton.cdiv(ncols, BLOCK_SIZE_N)
    col_idx = pid % triton.cdiv(ncols, BLOCK_SIZE_N)

    # Offsets for the current block
    row_start = row_idx * BLOCK_SIZE_M
    col_start = col_idx * BLOCK_SIZE_N

    # Create block pointers
    offs_m = row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for bounds checking
    mask = (offs_m[:, None] < nrows) & (offs_n[None, :] < ncols)

    # Compute memory offsets for matrices
    x_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    
    # Load data with mask
    x = tl.load(x_ptrs, mask=mask)
    y = tl.load(y_ptrs, mask=mask)

    # Compute SwiGLU: x * sigmoid(x) * y
    sigmoid_x = tl.sigmoid(x)
    result = x * sigmoid_x * y

    # Store the result
    out_ptrs = OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask)

def swiglu_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute SwiGLU forward pass: x * sigmoid(x) * y
    
    Args:
        x: Input tensor of shape (M, N)
        y: Input tensor of shape (M, N)
    Returns:
        Output tensor of shape (M, N)
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.dim() == 2, "Input tensors must be 2-dimensional"
    
    # Get tensor dimensions
    M, N = x.shape
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    y = y.contiguous()
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Calculate grid size
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
    
    # Launch kernel
    _swiglu_fwd_kernel[grid](
        output, x, y,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        output.stride(0), output.stride(1),
        M, N,
    )
    
    return output
