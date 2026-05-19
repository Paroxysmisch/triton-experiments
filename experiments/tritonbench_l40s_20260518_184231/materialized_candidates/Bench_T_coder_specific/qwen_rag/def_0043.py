import torch
import triton
import triton.language as tl

# Define the kernel for the matrix-vector product
@triton.jit
def _matmul_kernel(
    A_ptr, x_ptr, y_ptr, n: tl.int32, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(n):
        a = tl.load(A_ptr + row * n + k)
        xk = tl.load(x_ptr + k)
        acc += a * xk
    
    acc = tl.reduce(acc, axis=0, op=tl.sum)
    tl.store(y_ptr + row, acc)

# Define the kernel for calculating the norm
@triton.jit
def _norm_kernel(
    y_ptr, norm_ptr, n: tl.int32, p: tl.float32, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(n):
        yi = tl.load(y_ptr + i)
        acc += yi ** p
    
    acc = tl.reduce(acc, axis=0, op=tl.sum)
    norm = acc ** (1.0 / p)
    tl.store(norm_ptr + row, norm)

# Wrapper function for the symmetric matrix-vector product
def _symmetric_matrix_vector_product(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n = A.shape[0]
    y = torch.zeros_like(x, device=A.device, dtype=A.dtype)
    
    grid = (n,)
    block = (32,)
    _matmul_kernel[grid, block](
        A_ptr=A.data_ptr(),
        x_ptr=x.data_ptr(),
        y_ptr=y.data_ptr(),
        n=n,
        BLOCK_SIZE=block[0]
    )
    
    y *= alpha
    y += beta * x
    
    return y

# Wrapper function for calculating the norm
def _calculate_norm(y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    n = y.shape[0]
    norm = torch.zeros(1, device=y.device, dtype=y.dtype)
    
    grid = (n,)
    block = (32,)
    _norm_kernel[grid, block](
        y_ptr=y.data_ptr(),
        norm_ptr=norm.data_ptr(),
        n=n,
        p=p,
        BLOCK_SIZE=block[0]
    )
    
    return norm.item()

# Combined wrapper function
def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Matrix A must be square"
    assert x.ndim == 1 and x.shape[0] == A.shape[0], "Vector x must have the same dimension as the matrix A"
    
    y = _symmetric_matrix_vector_product(A, x, alpha, beta)
    norm = _calculate_norm(y, p)
    
    return norm

# Example usage
if __name__ == "__main__":
    A = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
    x = torch.tensor([1.0, 0.5], dtype=torch.float32)
    alpha = 1.0
    beta = 0.0
    p = 2.0
    
    norm = symmetric_matrix_vector_norm(A, x, alpha, beta, p)
    print(f"Norm: {norm}")
