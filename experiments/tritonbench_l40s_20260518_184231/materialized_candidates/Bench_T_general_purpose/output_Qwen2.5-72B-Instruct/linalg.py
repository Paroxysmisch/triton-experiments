import triton
import triton.language as tl

@triton.jit
def determinant_kernel(
    A_ptr,  # Pointer to the input matrix
    out_ptr,  # Pointer to the output tensor
    batch_size,  # Number of batches
    n,  # Size of the matrix (n x n)
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
    dtype: tl.constexpr  # Data type of the input matrix
):
    # Compute the determinant for each batch
    pid = tl.program_id(axis=0)
    if pid < batch_size:
        # Compute the determinant for the current batch
        A_batch_ptr = A_ptr + pid * n * n
        out_batch_ptr = out_ptr + pid

        # Initialize the determinant
        det = tl.zeros((1,), dtype=dtype)

        # Compute the determinant using LU decomposition
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    det *= A_batch_ptr[i * n + j]
                else:
                    factor = A_batch_ptr[j * n + i] / A_batch_ptr[i * n + i]
                    for k in range(i + 1, n):
                        A_batch_ptr[j * n + k] -= factor * A_batch_ptr[i * n + k]

        # Store the result
        tl.store(out_batch_ptr, det)

import torch
import triton
import triton.language as tl

def linalg_det(A, *, out=None):
    # Check input tensor shape and dtype
    if A.dim() < 2 or A.size(-1) != A.size(-2):
        raise ValueError("Input tensor must be a square matrix or a batch of square matrices.")
    
    # Determine the data type and block size
    dtype = A.dtype
    if dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Unsupported data type. Supported types are float, double, cfloat, and cdouble.")
    
    # Determine the block size
    BLOCK_SIZE = 16  # Adjust as needed

    # Determine the batch size and matrix size
    batch_size = A.size(0) if A.dim() > 2 else 1
    n = A.size(-1)

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(batch_size, dtype=dtype, device=A.device)

    # Launch the Triton kernel
    grid = (batch_size, )
    determinant_kernel[grid](
        A_ptr=A.data_ptr(),
        out_ptr=out.data_ptr(),
        batch_size=batch_size,
        n=n,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=dtype
    )

    return out
