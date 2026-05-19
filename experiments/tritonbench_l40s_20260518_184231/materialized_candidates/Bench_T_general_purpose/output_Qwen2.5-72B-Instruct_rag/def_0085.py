import torch
import triton
import triton.language as tl

# Triton kernel to compute the element-wise power of eigenvalues
@triton.jit
def power_kernel(eigenvalues_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    eigenvalues = tl.load(eigenvalues_ptr + offsets, mask=mask)
    powered_eigenvalues = tl.math.pow(eigenvalues, k)
    tl.store(eigenvalues_ptr + offsets, powered_eigenvalues, mask=mask)

# Wrapper function to compute the matrix power using eigendecomposition
def matrix_power_eig(A, k, *, out=None):
    # Check input dimensions
    if A.dim() < 2 or A.size(-1) != A.size(-2):
        raise ValueError("Input tensor must be a batch of square matrices.")
    
    # Compute the eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Ensure the eigenvalues are in a contiguous tensor
    eigenvalues = eigenvalues.contiguous()
    
    # Compute the element-wise power of eigenvalues using Triton
    n_elements = eigenvalues.numel()
    BLOCK_SIZE = 1024  # Block size for processing
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    power_kernel[grid](eigenvalues, k, n_elements, BLOCK_SIZE)
    
    # Reconstruct the matrix A^k = V diag(Λ^k) V^{-1}
    powered_matrix = torch.matmul(eigenvectors, torch.diag_embed(eigenvalues))
    powered_matrix = torch.matmul(powered_matrix, torch.linalg.inv(eigenvectors))
    
    # Handle the output tensor
    if out is not None:
        out.copy_(powered_matrix)
    else:
        out = powered_matrix
    
    return out

# Example usage
A = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
k = 2.0
result = matrix_power_eig(A, k)
print(result)
