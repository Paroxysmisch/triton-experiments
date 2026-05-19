import torch
import triton
import triton.language as tl


@triton.jit
def _solve_upper_triangular_kernel(
    A,
    b,
    y,
    n: tl.constexpr,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    block_row_idx = row_idx * block_size + tl.arange(0, block_size)

    # Load upper triangular matrix A in blocks
    A_block_ptr = A + block_row_idx[:, None] * n + block_row_idx[None, :]
    A_block = tl.load(A_block_ptr, mask=(block_row_idx[:, None] >= block_row_idx[None, :]), other=0.0)

    # Calculate y = inv(A) @ b using forward substitution
    y_hat = tl.sum(A_block.to(tl.float32) * b[block_row_idx], axis=1)
    y_hat = y_hat.to(b.dtype)

    # Residual computation
    residual_mask = block_row_idx < n
    residual = tl.where(residual_mask, b[block_row_idx] - y_hat, 0.0)

    # Store the result in y
    y_ptr = y + row_idx * block_size + tl.arange(0, block_size)
    tl.store(y_ptr, y_hat, mask=residual_mask)


def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Forward solves the linear system Ax=b where A is assumed to be an upper triangular matrix.
    Then adds a scaled vector y to the solution x.
    Args:
        A (torch.Tensor) : An upper triangular matrix of shape `(n, n)` .
        b (torch.Tensor) : Right hand side target vector/matrix of shape `(n,)` or `(n, k)` .
        y (torch.Tensor) : Vector to be scaled and added, must have shape `(n,)` or broadcastable to `(n,)` .
        alpha (float)    : Scaling factor for the vector `y`
    Returns:
        x (torch.Tensor) : Solution vector/matrix to Ax = b, has shape `(n,)` or `(n, k)`
    """
    n = A.size(-1)
    block_size = triton.next_power_of_2(n)

    # Flatten input tensors for batch processing
    b = b.reshape(-1, n)
    y = y.unsqueeze(0) if y.ndim == 1 else y
    y = y.reshape(-1, n)
    num_blocks = y.size(0)
    x = torch.empty_like(y)

    # Define the grid size for the kernel launch
    grid = (num_blocks, 1, 1)

    # Launch the Triton kernel for solving and adding vectors
    _solve_upper_triangular_kernel[grid](
        A,
        b,
        x,
        n,
        block_size,
    )

    # Scale the added vector and add it to the solution
    x += alpha * y

    # Reshape the output to match the input shape of y
    x = x.reshape_as(y)
    return x.squeeze(0) if y.ndim == 1 else x
