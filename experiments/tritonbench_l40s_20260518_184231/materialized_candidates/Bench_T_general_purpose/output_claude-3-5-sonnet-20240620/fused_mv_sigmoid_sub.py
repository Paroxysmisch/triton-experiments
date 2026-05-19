import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(A, v, b, alpha, out, n, m):
    # Compute the matrix-vector multiplication
    row = tl.program_id(0)
    if row < n:
        z = 0.0
        for j in range(m):
            z += A[row, j] * v[j]
        
        # Sigmoid activation
        s = 1 / (1 + tl.exp(-z))
        
        # Subtract alpha * b
        out[row] = s - alpha * b

import torch

def fused_mv_sigmoid_sub(input: torch.Tensor, vec: torch.Tensor, other: torch.Tensor, alpha: float = 1, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure input shapes are compatible
    n, m = input.shape
    assert vec.shape == (m,), "Vector shape must match the number of columns in input matrix."
    assert other.shape == () or other.shape == (n,), "Other must be a scalar or broadcastable to the output shape."

    # Prepare output tensor
    if out is None:
        out = torch.empty(n, dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    grid = (n,)
    fused_mv_sigmoid_sub_kernel[grid](input, vec, other, alpha, out, n, m)

    return out
