import triton
import triton.language as tl
import torch

# Triton kernel to compute the spectral norm of a matrix using its eigenvalues
@triton.jit
def spectral_norm_eig_kernel(
    A_ptr,  # Pointer to the input matrix
    A_batch_stride,  # Stride between batches in the input matrix
    A_row_stride,  # Stride between rows in the input matrix
    A_col_stride,  # Stride between columns in the input matrix
    out_ptr,  # Pointer to the output tensor
    n,  # Size of the square matrix
    batch_size: tl.constexpr,  # Number of batches
    BLOCK_SIZE: tl.constexpr  # Block size for parallelism
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # Compute the eigenvalues of the matrix
    A_start_ptr = A_ptr + batch_idx * A_batch_stride
    eigenvalues = tl.eig(A_start_ptr, A_row_stride, A_col_stride, n)

    # Compute the absolute values of the eigenvalues
    abs_eigenvalues = tl.abs(eigenvalues)

    # Find the maximum absolute eigenvalue
    max_abs_eigenvalue = tl.max(abs_eigenvalues, axis=0)

    # Store the result in the output tensor
    out_start_ptr = out_ptr + batch_idx
    tl.store(out_start_ptr, max_abs_eigenvalue)

# Wrapper function to call the Triton kernel
def spectral_norm_eig(A, *, out=None):
    # Validate input tensor
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Input tensor must be a batch of square matrices")

    # Determine the number of batches and the size of the square matrix
    batch_size = A.shape[:-2].numel()
    n = A.shape[-1]

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(batch_size, device=A.device, dtype=A.dtype)

    # Determine the block size for parallelism
    BLOCK_SIZE = 128  # Adjust this value based on the GPU architecture

    # Enqueue the kernel
    spectral_norm_eig_kernel[(batch_size,)](
        A,
        A.stride(0) * A.element_size(),
        A.stride(1) * A.element_size(),
        A.stride(2) * A.element_size(),
        out,
        n,
        batch_size,
        num_warps=4,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# Example usage of the kernel
torch.manual_seed(0)
A = torch.randn(2, 3, 3, device='cuda', dtype=torch.float32)
out = spectral_norm_eig(A)
print(out)
