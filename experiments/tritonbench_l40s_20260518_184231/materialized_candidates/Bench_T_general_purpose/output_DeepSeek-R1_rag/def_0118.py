import triton
import triton.language as tl
import torch

@triton.jit
def add_scaled_vector_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n_rows,
    n_cols,
    x_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_numel
    row = offsets // n_cols
    y = tl.load(y_ptr + row, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_updated = x + alpha * y
    tl.store(x_ptr + offsets, x_updated, mask=mask)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Solve the triangular system
    x = torch.linalg.solve_triangular(A, b, upper=True)
    n = A.size(-2)
    
    # Ensure y is broadcastable to (n,)
    try:
        y_broadcasted = y.broadcast_to((n,))
    except RuntimeError:
        raise ValueError("y must be broadcastable to shape (n,)")
    
    # Reshape x to 2D if necessary and prepare for kernel
    if x.dim() == 1:
        x = x.unsqueeze(1)
        n_cols = 1
    else:
        n_cols = x.size(1)
    
    # Expand y to (n, 1) for column-wise broadcasting and flatten
    y_expanded = y_broadcasted.unsqueeze(1)
    x_flat = x.view(-1)
    y_flat = y_expanded.view(-1)
    x_numel = x_flat.numel()
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(x_numel, meta['BLOCK_SIZE']),)
    add_scaled_vector_kernel[grid](
        x_flat, y_flat, alpha, n, n_cols, x_numel, BLOCK_SIZE=1024
    )
    
    # Reshape x back to original shape if necessary
    if n_cols == 1:
        x = x.squeeze(1)
    
    return x
