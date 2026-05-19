import triton
import triton.language as tl
import torch

@triton.jit
def _add_scaled_vector_kernel(
    x_ptr,      # Pointer to solution x
    y_ptr,      # Pointer to vector y
    alpha,      # Scaling factor
    N,          # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out_val = x_val + alpha * y_val

    tl.store(x_ptr + offsets, out_val, mask=mask)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Solves the triangular system Ax = b for an upper triangular matrix A,
    then adds alpha * y to the solution x.
    """
    # Solve Ax = b using PyTorch's solve_triangular
    x = torch.linalg.solve_triangular(A, b, upper=True)

    # Broadcast y if needed
    y_b = y.expand_as(x)

    # Ensure x and y are on the same device and contiguous
    x = x.contiguous()
    y_b = y_b.contiguous()

    # Set up the Triton kernel for adding alpha * y to x
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _add_scaled_vector_kernel[grid](
        x, y_b, alpha, N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return x
