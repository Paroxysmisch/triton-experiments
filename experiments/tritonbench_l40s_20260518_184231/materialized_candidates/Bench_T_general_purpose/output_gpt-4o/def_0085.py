import torch
import triton
import triton.language as tl

# Triton kernel for computing matrix power using eigendecomposition
@triton.jit
def matrix_power_kernel(V_ptr, Lambda_ptr, Vinv_ptr, k, out_ptr, n, batch_size):
    # Obtain the batch index
    batch_idx = tl.program_id(0)

    # Load the eigenvectors and eigenvalues for the current batch
    V = tl.load(V_ptr + batch_idx * n * n, shape=(n, n))
    Lambda = tl.load(Lambda_ptr + batch_idx * n, shape=(n,))
    Vinv = tl.load(Vinv_ptr + batch_idx * n * n, shape=(n, n))

    # Compute Λ^k
    Lambda_k = Lambda ** k

    # Form the diagonal matrix of Λ^k
    Lambda_k_diag = tl.diag(Lambda_k)

    # Compute A^k = V * diag(Λ^k) * V^(-1)
    result = tl.dot(V, tl.dot(Lambda_k_diag, Vinv))

    # Store the result
    tl.store(out_ptr + batch_idx * n * n, result)

# Wrapper function
def matrix_power_eig(A, k, *, out=None):
    # Ensure input is a tensor
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")

    # Check if A is a square matrix
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a batch of square matrices")

    # Get dimensions
    *batch_dims, n, _ = A.shape
    batch_size = torch.prod(torch.tensor(batch_dims)).item()

    # Compute the eigenvalues and eigenvectors
    Lambda, V = torch.linalg.eig(A)

    # Compute the inverse of V
    Vinv = torch.linalg.inv(V)

    # Prepare output tensor
    if out is None:
        out = torch.empty_like(A, dtype=torch.complex128 if torch.is_complex(A) else torch.float64)

    # Launch Triton kernel
    grid = (batch_size,)
    matrix_power_kernel[grid](V, Lambda, Vinv, k, out, n, batch_size)

    return out

# Example usage
A = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
k = 2.0
result = matrix_power_eig(A, k)
print(result)
