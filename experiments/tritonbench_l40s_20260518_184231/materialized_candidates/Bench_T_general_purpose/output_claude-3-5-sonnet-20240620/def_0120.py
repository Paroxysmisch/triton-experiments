import torch
import triton
import triton.language as tl

@triton.jit
def matrix_vector_dot_kernel(
    # Pointers to matrices
    a_ptr, x_ptr, y_ptr,
    # Matrix dimensions
    M, N,
    # Scalar values
    alpha, beta,
    # Strides
    stride_am, stride_an,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Matrix multiplication part
    pid = tl.program_id(0)
    
    # Each program handles one row
    row = pid
    
    if row >= M:
        return
        
    # Compute the dot product for this row
    acc = 0.0
    for n in range(0, N, BLOCK_SIZE):
        # Load chunks of the row and x vector
        mask = n + tl.arange(0, BLOCK_SIZE) < N
        a = tl.load(a_ptr + row * stride_am + n * stride_an, mask=mask, other=0.0)
        x = tl.load(x_ptr + n, mask=mask, other=0.0)
        acc += tl.sum(a * x, axis=0)
    
    # Update y with alpha * (A @ x) + beta * y
    y_val = tl.load(y_ptr + row)
    new_y = alpha * acc + beta * y_val
    tl.store(y_ptr + row, new_y)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, 
                     alpha: float, beta: float) -> torch.Tensor:
    """
    Computes y = alpha * (A @ x) + beta * y, then returns dot(y, x)
    
    Args:
        A (Tensor): Input matrix of shape (n, m)
        x (Tensor): Input vector of shape (m,)
        y (Tensor): Target vector of shape (n,) to be modified in-place
        alpha (float): Scalar multiplier for A @ x
        beta (float): Scalar multiplier for y
    
    Returns:
        Tensor: The dot product of the updated y with x
    """
    assert A.dim() == 2, "Matrix A must be 2-dimensional"
    assert x.dim() == 1, "Vector x must be 1-dimensional"
    assert y.dim() == 1, "Vector y must be 1-dimensional"
    
    M, N = A.shape
    assert x.shape[0] == N, f"x shape {x.shape[0]} must match A's columns {N}"
    assert y.shape[0] == M, f"y shape {y.shape[0]} must match A's rows {M}"
    
    # Ensure contiguous tensors
    A = A.contiguous()
    x = x.contiguous()
    y = y.contiguous()
    
    # Configure the kernel
    BLOCK_SIZE = 32
    
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
    matrix_vector_dot_kernel[grid](
        A.data_ptr(), x.data_ptr(), y.data_ptr(),
        M, N,
        alpha, beta,
        A.stride(0), A.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Compute the final dot product
    return torch.dot(y, x)
