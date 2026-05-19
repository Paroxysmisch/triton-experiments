import torch
import triton
import triton.language as tl

@triton.jit
def solve_and_add_scaled_vector_kernel(
    # Pointers to matrices
    A_ptr, b_ptr, y_ptr, x_out_ptr,
    # Matrix dimensions
    n: tl.constexpr,
    # Strides
    A_stride_row, A_stride_col,
    b_stride, y_stride,
    # Other parameters
    alpha: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Handle one row per thread
    row = pid
    
    if row >= n:
        return
        
    # Load b[row]
    b_val = tl.load(b_ptr + row * b_stride)
    
    # Initialize x[row]
    x_val = b_val
    
    # Backward substitution for upper triangular system
    for col in range(n-1, row-1, -1):
        if col > row:
            # Load A[row,col] and x[col]
            A_val = tl.load(A_ptr + row * A_stride_row + col * A_stride_col)
            x_col = tl.load(x_out_ptr + col)
            x_val -= A_val * x_col
            
    # Divide by diagonal element
    A_diag = tl.load(A_ptr + row * A_stride_row + row * A_stride_col)
    x_val = x_val / A_diag
    
    # Load and add scaled y vector
    y_val = tl.load(y_ptr + row * y_stride)
    x_val = x_val + alpha * y_val
    
    # Store result
    tl.store(x_out_ptr + row, x_val)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Solves the triangular system Ax = b and adds a scaled vector y.
    
    Args:
        A (Tensor): Upper triangular matrix of shape (n, n)
        b (Tensor): Right-hand side vector of shape (n,) or (n, k)
        y (Tensor): Vector to be scaled and added of shape (n,)
        alpha (float): Scaling factor for vector y
        
    Returns:
        Tensor: Solution vector x of shape (n,)
    """
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    assert b.size(0) == A.size(0), "b must have compatible dimensions with A"
    assert y.size(0) == A.size(0), "y must have compatible dimensions with A"
    
    n = A.size(0)
    
    # Initialize output tensor
    x = torch.empty_like(b)
    
    # Configure grid and block sizes
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    solve_and_add_scaled_vector_kernel[grid](
        A_ptr=A.data_ptr(),
        b_ptr=b.data_ptr(),
        y_ptr=y.data_ptr(),
        x_out_ptr=x.data_ptr(),
        n=n,
        A_stride_row=A.stride(0),
        A_stride_col=A.stride(1),
        b_stride=b.stride(0),
        y_stride=y.stride(0),
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x
