import triton
import triton.language as tl

@triton.jit
def qr_kernel(
    A_ptr, Q_ptr, R_ptr, 
    m, n, 
    batch_size, 
    mode, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    batch_idx = pid // (m * n)
    mat_idx = pid % (m * n)

    if batch_idx < batch_size:
        A = tl.load(A_ptr + batch_idx * m * n + mat_idx * BLOCK_SIZE_M * BLOCK_SIZE_N, mask=mat_idx < m * n, other=0.0)
        
        if mode == 'reduced':
            Q, R = tl.qr(A, mode='reduced')
        elif mode == 'complete':
            Q, R = tl.qr(A, mode='complete')
        elif mode == 'r':
            R = tl.qr(A, mode='r')
            Q = tl.empty((0, 0), dtype=tl.float32)  # Empty tensor for Q

        tl.store(Q_ptr + batch_idx * m * n + mat_idx * BLOCK_SIZE_M * BLOCK_SIZE_N, Q, mask=mat_idx < m * n)
        tl.store(R_ptr + batch_idx * m * n + mat_idx * BLOCK_SIZE_M * BLOCK_SIZE_N, R, mask=mat_idx < m * n)

import torch
import triton
import triton.language as tl

def qr(A, mode='reduced', *, out=None):
    # Validate input
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    m, n = A.shape[-2:]
    batch_size = A.shape[:-2] if A.dim() > 2 else 1
    batch_size = int(torch.prod(torch.tensor(batch_size)))

    # Determine the shape of Q and R based on mode
    if mode == 'reduced':
        Q_shape = (*A.shape[:-2], m, n)
        R_shape = (*A.shape[:-2], n, n)
    elif mode == 'complete':
        Q_shape = (*A.shape[:-2], m, m)
        R_shape = (*A.shape[:-2], m, n)
    elif mode == 'r':
        Q_shape = (*A.shape[:-2], 0, 0)  # Empty tensor for Q
        R_shape = (*A.shape[:-2], n, n)
    else:
        raise ValueError("mode must be one of 'reduced', 'complete', or 'r'")

    # Allocate output tensors
    if out is None:
        Q = torch.empty(Q_shape, dtype=A.dtype, device=A.device)
        R = torch.empty(R_shape, dtype=A.dtype, device=A.device)
    else:
        Q, R = out
        if Q.shape != Q_shape or R.shape != R_shape or Q.dtype != A.dtype or R.dtype != A.dtype:
            raise ValueError("Output tensors have incorrect shape or dtype")

    # Launch the kernel
    grid = (batch_size * m * n // (BLOCK_SIZE_M * BLOCK_SIZE_N),)
    qr_kernel[grid](
        A, Q, R, 
        m, n, 
        batch_size, 
        mode, 
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return Q, R

# Constants for block sizes
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16
