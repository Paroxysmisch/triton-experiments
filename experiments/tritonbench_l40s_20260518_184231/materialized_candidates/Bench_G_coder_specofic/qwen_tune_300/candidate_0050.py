import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(X, N, BLOCK_SIZE: tl.constexpr):
    # Triton kernel to perform ReLU on input tensor X
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    x = tl.where(x > 0, x, 0)
    if pid == 0:
        tl.store(X + offsets, x, mask=mask)

def relu(X: torch.Tensor) -> torch.Tensor:
    # Wrapper function to call the Triton kernel for ReLU
    assert X.is_contiguous()
    N = X.numel()
    BLOCK_SIZE = 512
    grid = triton.cdiv(N, BLOCK_SIZE)
    relu_kernel[grid,](X, N, BLOCK_SIZE)
    return X
