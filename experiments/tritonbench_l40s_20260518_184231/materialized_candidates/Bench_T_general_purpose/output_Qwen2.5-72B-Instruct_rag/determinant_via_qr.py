import torch
import triton
import triton.language as tl

@triton.jit
def determinant_kernel(A_ptr, Q_ptr, R_ptr, det_ptr, n, BLOCK_SIZE: tl.constexpr):
    # The block size is the smallest power of two greater than the number of columns in `A`
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Load the matrix A into SRAM
    A = tl.load(A_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n, other=0.0)

    # Perform QR decomposition
    Q, R = torch.linalg.qr(A)

    # Store Q and R matrices
    tl.store(Q_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), Q, mask=tl.arange(0, BLOCK_SIZE) < n)
    tl.store(R_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), R, mask=tl.arange(0, BLOCK_SIZE) < n)

    # Compute the determinant of Q
    det_Q = torch.det(Q)

    # Compute the determinant of R
    det_R = 1.0
    for i in range(n):
        det_R *= R[i, i]

    # Compute the determinant of A
    det_A = det_Q * det_R

    # Store the determinant
    tl.store(det_ptr + block_start, det_A, mask=block_start < n)

import torch

def determinant_via_qr(A, *, mode='reduced', out=None):
    # Ensure A is a square matrix
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    n = A.size(0)

    # Allocate memory for Q, R, and det
    Q = torch.empty_like(A)
    R = torch.empty_like(A)
    det = torch.empty(1, device=A.device, dtype=A.dtype)

    # Define the block size
    BLOCK_SIZE = triton.next_power_of_2(n)

    # Launch the Triton kernel
    determinant_kernel[(1,)](
        A, Q, R, det, n,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Return the determinant
    if out is not None:
        out.copy_(det)
        return out
    else:
        return det
