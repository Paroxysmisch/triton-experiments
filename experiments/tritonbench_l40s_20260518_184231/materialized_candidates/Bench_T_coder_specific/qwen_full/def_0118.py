import torch
import triton
import triton.language as tl

@triton.jit
def solve_and_add_scaled_vector_tri(A, b, y, n, k, alpha):
    # Triton kernel to solve triangular system and add scaled vector
    pid = tl.program_id(0)
    if k == 1:
        b = b.reshape(n)
        y = y.reshape(n)
        x = tl.arange(0, n)
        mask = x < n
        a = tl.load(A + x)
        c = tl.load(b + x, mask=mask)
        x = tl.where(mask, c / a, 0.0)
        x = x.to(tl.float32)
        y = y.to(tl.float32)
        x = x + alpha * y
        tl.store(b + x, x, mask=mask)
    else:
        b = b.reshape(n, k)
        y = y.reshape(n, 1)
        x = tl.arange(0, n)
        mask = x[:, None] < n
        a = tl.load(A + x * k + (k - 1), mask=mask)
        c = tl.load(b + x * k, mask=mask)
        x = tl.where(mask, c / a, 0.0)
        x = x.to(tl.float32)
        y = y.to(tl.float32)
        x = x + alpha * y
        tl.store(b + x * k, x, mask=mask)

def solve_and_add_scaled_vector_triton(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Wrapper function for Triton kernel
    n, k = b.shape
    grid = (triton.cdiv(n, 1),)
    if k == 1:
        y = y.reshape(n)
    else:
        y = y.reshape(n, 1)
    solve_and_add_scaled_vector_tri[grid](A, b, y, n, k, alpha)
    return b
