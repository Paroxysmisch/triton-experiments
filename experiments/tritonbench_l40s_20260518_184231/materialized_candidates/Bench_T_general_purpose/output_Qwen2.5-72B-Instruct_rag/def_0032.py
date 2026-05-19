import torch
import triton
import triton.language as tl
import torch.cuda as cuda
import torch.linalg as linalg

# Triton kernel for eigenvalue decomposition
@triton.jit
def eig_kernel(A_ptr, eigenvalues_ptr, eigenvectors_ptr, n, batch_size, BLOCK_SIZE: tl.constexpr):
    # Get the batch index
    batch_idx = tl.program_id(0)
    if batch_idx < batch_size:
        # Compute the offset for the current batch
        A_offset = batch_idx * n * n
        eigenvalues_offset = batch_idx * n
        eigenvectors_offset = batch_idx * n * n

        # Load the matrix A
        A = tl.load(A_ptr + A_offset, (n, n))

        # Compute eigenvalues and eigenvectors using cuSOLVER
        eigenvalues, eigenvectors = linalg.eig(A)

        # Store the results
        tl.store(eigenvalues_ptr + eigenvalues_offset, eigenvalues)
        tl.store(eigenvectors_ptr + eigenvectors_offset, eigenvectors)

# Wrapper function for eigenvalue decomposition
def linalg_eig(A, *, out=None):
    # Check input tensor shape and dtype
    if A.dim() < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError("Input tensor must be of shape (*, n, n) where * is zero or more batch dimensions.")
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Input tensor must have dtype float, double, cfloat, or cdouble.")

    # Determine batch size and matrix size
    batch_size = A.shape[:-2] if A.dim() > 2 else 1
    n = A.shape[-1]

    # Allocate output tensors if not provided
    if out is None:
        eigenvalues = torch.empty(batch_size + (n,), dtype=A.dtype, device=A.device)
        eigenvectors = torch.empty(batch_size + (n, n), dtype=A.dtype, device=A.device)
    else:
        eigenvalues, eigenvectors = out
        if eigenvalues.shape != batch_size + (n,) or eigenvectors.shape != batch_size + (n, n):
            raise ValueError("Output tensors must have the correct shape.")
        if eigenvalues.dtype != A.dtype or eigenvectors.dtype != A.dtype:
            raise ValueError("Output tensors must have the same dtype as the input tensor.")

    # Determine block size
    BLOCK_SIZE = 128

    # Launch the kernel
    grid = (batch_size, 1, 1)
    eig_kernel[grid](A, eigenvalues, eigenvectors, n, batch_size, BLOCK_SIZE=BLOCK_SIZE)

    # Synchronize the device with the CPU
    cuda.synchronize()

    return eigenvalues, eigenvectors

# Example usage
if __name__ == "__main__":
    A = torch.randn(2, 3, 3, dtype=torch.float32, device="cuda")
    eigenvalues, eigenvectors = linalg_eig(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)
