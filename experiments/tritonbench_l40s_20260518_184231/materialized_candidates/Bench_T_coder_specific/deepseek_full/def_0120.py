import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def matrix_vector_dot(A: tl.tensor, x: tl.tensor, y: tl.tensor, alpha: tl.tensor, beta: tl.tensor) -> tl.tensor:
    # Compute y = alpha * Ax + beta * y
    tl.store(y, tl.load(y) * beta + alpha * tl.dot(A, x))
    # Return the dot product y . x
    return tl.dot(y, x)

def test_matrix_vector_dot() -> None:
    # Create some random data
    n, m = 64, 64
    A = torch.randn((n, m), device="cuda")
    x = torch.randn((m,), device="cuda")
    y = torch.randn((n,), device="cuda")
    alpha, beta = 2.0, 3.0
    # Call the Triton kernel
    result = matrix_vector_dot(A, x, y, alpha, beta)
    # Compute the result using torch
    y = torch.mv(A, x) * alpha + torch.mv(A, y) * beta
    expected = torch.dot(y, x)
    # Assert that the result is close
    assert torch.allclose(result, expected)
