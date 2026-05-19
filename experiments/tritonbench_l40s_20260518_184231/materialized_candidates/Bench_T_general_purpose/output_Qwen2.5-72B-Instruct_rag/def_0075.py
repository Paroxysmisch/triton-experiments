import torch
import triton
import triton.language as tl

# Define the fused Cholesky solve function
def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ensure A is a symmetric positive-definite matrix
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == b.shape[0], "A and b must have the same number of rows"
    
    # Perform Cholesky decomposition: A = L * L.T
    L = torch.cholesky(A, upper=False)
    
    # Solve the system L * y = b for y
    y = torch.triangular_solve(b, L, upper=False).solution
    
    # Solve the system L.T * x = y for x
    x = torch.triangular_solve(y, L.T, upper=True).solution
    
    return x

# Example usage
if __name__ == "__main__":
    # Create a symmetric positive-definite matrix A and a right-hand side tensor b
    A = torch.tensor([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=torch.float32)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    
    # Solve the system Ax = b
    x = fused_cholesky_solve(A, b)
    print("Solution x:")
    print(x)
