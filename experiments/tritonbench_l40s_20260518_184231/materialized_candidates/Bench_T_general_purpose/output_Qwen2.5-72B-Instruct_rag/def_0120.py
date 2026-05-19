import torch
import triton
import triton.language as tl

# Triton kernel for matrix-vector product and update
@triton.jit
def matrix_vector_product_kernel(
    A_ptr, x_ptr, y_ptr, alpha, beta,
    M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    y_offsets = offsets

    # Load y into a vector
    y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for n in range(0, N, BLOCK_SIZE):
        a_offsets = pid * N * BLOCK_SIZE + n + tl.arange(0, BLOCK_SIZE)
        x_offsets = n + tl.arange(0, BLOCK_SIZE)
        a_mask = a_offsets < M * N
        x_mask = x_offsets < N

        # Load A and x
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # Perform the matrix-vector product
        acc += a * x

    # Scale the accumulator by alpha
    acc *= alpha

    # Update y with the scaled product and beta * y
    y = acc + beta * y

    # Store the updated y
    tl.store(y_ptr + y_offsets, y, mask=mask)

# Python wrapper function
def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2, "A must be a 2D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.dim() == 1, "y must be a 1D tensor"
    assert A.shape[1] == x.shape[0], "A's second dimension must match x's first dimension"
    assert A.shape[0] == y.shape[0], "A's first dimension must match y's first dimension"

    M, N = A.shape
    BLOCK_SIZE = 128  # Adjust block size as needed

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    matrix_vector_product_kernel[grid](
        A, x, y, alpha, beta, M, N, BLOCK_SIZE
    )

    # Compute the dot product of the updated y with x
    result = torch.dot(y, x)
    return result
