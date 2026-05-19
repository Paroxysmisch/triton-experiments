import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(
    A_ptr,  # Pointer to the input matrix
    L_ptr,  # Pointer to the output matrix
    n,      # Size of the matrix
    batch_size,  # Batch size
    upper,  # Whether to return an upper triangular matrix
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
    TYPE: tl.constexpr  # Data type of the matrix elements
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (n // BLOCK_SIZE)
    block_id = pid % (n // BLOCK_SIZE)

    # Compute the block of the matrix
    block_start = block_id * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n)

    # Load the block of the matrix
    A_block = tl.load(A_ptr + batch_id * n * n + block_start * n + block_start, mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)

    # Perform the Cholesky decomposition on the block
    for i in range(block_end):
        for j in range(i, block_end):
            if i == j:
                A_block[i, j] = tl.sqrt(A_block[i, j] - tl.sum(A_block[i, :i] * A_block[i, :i]))
            else:
                A_block[j, i] = (A_block[j, i] - tl.sum(A_block[j, :i] * A_block[i, :i])) / A_block[i, i]

    # Store the result
    if upper:
        L_block = tl.transpose(tl.conj(A_block))
    else:
        L_block = A_block

    tl.store(L_ptr + batch_id * n * n + block_start * n + block_start, L_block, mask=block_start + tl.arange(0, BLOCK_SIZE) < n)

import torch
import triton

def linalg_cholesky(A, *, upper=False, out=None):
    # Validate input
    if A.dim() < 2 or A.size(-1) != A.size(-2):
        raise RuntimeError("Input tensor must be a square matrix or a batch of square matrices.")
    
    # Determine the data type
    dtype = A.dtype
    if dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise RuntimeError("Input tensor must have dtype float, double, cfloat, or cdouble.")
    
    # Determine the batch size
    batch_size = 1 if A.dim() == 2 else A.size(0)
    n = A.size(-1)
    
    # Allocate output tensor
    if out is None:
        out = torch.empty_like(A, dtype=dtype, device=A.device)
    else:
        if out.shape != A.shape or out.dtype != A.dtype or out.device != A.device:
            raise RuntimeError("Output tensor must have the same shape, dtype, and device as the input tensor.")
    
    # Define the block size
    BLOCK_SIZE = 32
    
    # Launch the Triton kernel
    grid = (batch_size * (n // BLOCK_SIZE),)
    cholesky_kernel[grid](
        A, out, n, batch_size, upper, BLOCK_SIZE, dtype
    )
    
    # Synchronize the device with the CPU
    if A.device.type == 'cuda':
        torch.cuda.synchronize(A.device)
    
    return out

# Example usage
A = torch.tensor([[[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]], dtype=torch.float64, device='cuda')
L = linalg_cholesky(A, upper=False)
print(L)
