import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mv_norm_kernel(
    # Pointers to matrices
    a_ptr, x_ptr, y_ptr,
    # Matrix dimensions
    n,
    # Strides
    stride_am, stride_an,
    stride_x, stride_y,
    # Constants
    alpha, beta,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Row index
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < n
    
    # Initialize accumulator for this row
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over columns
    for k in range(0, n, BLOCK_SIZE):
        # Load x values
        k_idx = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_idx < n
        x_vals = tl.load(x_ptr + k_idx * stride_x, mask=k_mask, other=0.0)
        
        # Load matrix values
        a_vals = tl.load(
            a_ptr + row[:, None] * stride_am + k_idx[None, :] * stride_an,
            mask=mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Accumulate matrix-vector product
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)
    
    # Load existing y values if beta != 0
    y_old = tl.load(y_ptr + row * stride_y, mask=mask, other=0.0)
    
    # Compute final result with alpha and beta
    y_new = alpha * acc + beta * y_old
    
    # Store result
    tl.store(y_ptr + row * stride_y, y_new, mask=mask)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    # Input validation
    assert A.dim() == 2 and A.size(0) == A.size(1), "Matrix A must be square"
    assert x.dim() == 1 and x.size(0) == A.size(1), "Vector x dimensions must match matrix A"
    
    # Get dimensions
    n = A.size(0)
    
    # Create output vector
    y = torch.zeros_like(x)
    
    # Configure grid and block sizes
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    symmetric_mv_norm_kernel[grid](
        A, x, y,
        n,
        A.stride(0), A.stride(1),
        x.stride(0), y.stride(0),
        alpha, beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Compute norm
    return torch.norm(y, p=p)
