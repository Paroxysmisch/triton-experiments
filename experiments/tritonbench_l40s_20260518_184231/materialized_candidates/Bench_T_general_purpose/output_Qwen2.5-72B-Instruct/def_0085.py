import triton
import triton.language as tl

@triton.jit
def matrix_power_eig_kernel(
    A_ptr,  # Pointer to the input matrix A
    k,      # The exponent to which the matrix A is to be raised
    out_ptr,  # Pointer to the output tensor
    n,       # Size of the matrix (n x n)
    batch_size,  # Batch size
    BLOCK_SIZE: tl.constexpr
):
    # Compute the batch index
    pid = tl.program_id(axis=0)
    batch_idx = pid // (n * n)
    if batch_idx >= batch_size:
        return

    # Compute the row and column indices
    row_idx = (pid % (n * n)) // n
    col_idx = (pid % (n * n)) % n

    # Load the matrix A
    A = tl.load(A_ptr + batch_idx * n * n + row_idx * n + col_idx)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = tl.linalg.eig(A)

    # Compute the diagonal matrix of eigenvalues raised to the power k
    lambda_k = tl.where(eigenvalues != 0, eigenvalues ** k, 0.0)

    # Compute the matrix power using the formula A^k = V diag(Λ^k) V^{-1}
    V = eigenvectors
    V_inv = tl.linalg.inv(eigenvectors)
    A_k = tl.matmul(tl.matmul(V, tl.diag(lambda_k)), V_inv)

    # Store the result in the output tensor
    tl.store(out_ptr + batch_idx * n * n + row_idx * n + col_idx, A_k)

import torch
import triton
import triton.language as tl

def matrix_power_eig(A, k, *, out=None) -> torch.Tensor:
    # Check input tensor properties
    if A.dim() < 2 or A.size(-1) != A.size(-2):
        raise ValueError("Input tensor must be a batch of square matrices.")
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(A, dtype=A.dtype, device=A.device)
    else:
        if out.shape != A.shape or out.dtype != A.dtype or out.device != A.device:
            raise ValueError("Output tensor must have the same shape, dtype, and device as the input tensor.")
    
    # Get the batch size and matrix size
    batch_size = A.shape[:-2] if A.dim() > 2 else 1
    n = A.size(-1)
    
    # Flatten the batch dimensions for the kernel
    A_flat = A.view(-1, n, n)
    out_flat = out.view(-1, n, n)
    
    # Launch the Triton kernel
    grid = (A_flat.numel(),)
    matrix_power_eig_kernel[grid](
        A_flat, k, out_flat, n, batch_size, BLOCK_SIZE=16
    )
    
    return out
