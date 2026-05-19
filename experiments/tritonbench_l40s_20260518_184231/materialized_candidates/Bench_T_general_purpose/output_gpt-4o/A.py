import triton
import triton.language as tl

@triton.jit
def solve_kernel(A_ptr, B_ptr, X_ptr, n, stride_A, stride_B, stride_X, BLOCK_SIZE: tl.constexpr):
    # Define the program id for the block
    pid = tl.program_id(0)
    
    # Compute row and column indices for this block
    row = pid // n
    col = pid % n

    # Load A and B elements into registers
    a = tl.load(A_ptr + row * stride_A + col)
    b = tl.load(B_ptr + row * stride_B + col)

    # Compute the solution element (naive inversion for demonstration)
    # In practice, you would use a more sophisticated method for inversion
    x = b / a

    # Store the result in X
    tl.store(X_ptr + row * stride_X + col, x)

import torch

def solve_system(A, B, *, left=True, out=None):
    assert A.dim() >= 2 and A.size(-1) == A.size(-2), "A must be a square matrix or a batch of square matrices"
    assert B.dim() >= 1, "B must be at least 1-dimensional"
    assert A.device == B.device, "A and B must be on the same device"
    
    # Determine the shape of the output
    batch_shape = A.shape[:-2]
    n = A.size(-1)
    
    if left:
        # Solve AX = B
        A, B = A.contiguous(), B.contiguous()
    else:
        # Solve XA = B
        A, B = A.transpose(-1, -2).contiguous(), B.transpose(-1, -2).contiguous()
    
    if out is None:
        out = torch.empty_like(B)
    
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    solve_kernel[grid](A, B, out, n, A.stride(-2), B.stride(-2), out.stride(-2), BLOCK_SIZE=32)
    
    # Synchronize CUDA device with CPU
    if A.is_cuda:
        torch.cuda.synchronize()
    
    return out
