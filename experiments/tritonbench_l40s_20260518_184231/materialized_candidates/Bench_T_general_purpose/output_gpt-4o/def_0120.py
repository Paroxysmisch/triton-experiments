import torch
import triton
import triton.language as tl

# Triton kernel for scaled matrix-vector product and updating y
@triton.jit
def matrix_vector_update_kernel(A_ptr, x_ptr, y_ptr, alpha, beta, n, m, BLOCK_SIZE: tl.constexpr):
    # Pointers to the start of the row in A and corresponding elements in x and y
    row_idx = tl.program_id(0)
    row_start = row_idx * m

    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the columns of A and elements of x
    for col_idx in range(0, m, BLOCK_SIZE):
        # Load a block of A and x
        a = tl.load(A_ptr + row_start + col_idx + tl.arange(0, BLOCK_SIZE))
        x = tl.load(x_ptr + col_idx + tl.arange(0, BLOCK_SIZE))
        
        # Accumulate the product
        acc += a * x

    # Sum up the accumulated results
    result = tl.sum(acc, axis=0)

    # Load y, scale and update it
    y = tl.load(y_ptr + row_idx)
    y_new = alpha * result + beta * y

    # Store the updated y
    tl.store(y_ptr + row_idx, y_new)

# Wrapper function
def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    assert x.shape == (m,)
    assert y.shape == (n,)

    # Launch the Triton kernel
    BLOCK_SIZE = 128  # Adjust this based on your GPU capabilities
    grid = (n,)  # One block per row of A
    matrix_vector_update_kernel[grid](A, x, y, alpha, beta, n, m, BLOCK_SIZE)

    # Compute the dot product using PyTorch
    result = torch.dot(y, x)
    
    return result
