import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(X, Y, OUT, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    # Compute the sigmoid of x
    sigmoid_x = 1 / (1 + tl.exp(-x))

    # Compute the element-wise multiplication
    out = sigmoid_x * x * y

    tl.store(OUT + offsets, out, mask=mask)

import torch
from triton.runtime import launch_kernel

def _swiglu_fwd(x, y, out):
    assert x.is_contiguous(), "Input x must be contiguous"
    assert y.is_contiguous(), "Input y must be contiguous"
    assert out.is_contiguous(), "Output out must be contiguous"
    assert x.shape == y.shape, "Input x and y must have the same shape"
    assert out.shape == x.shape, "Output out must have the same shape as input x"

    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # Convert tensors to Triton pointers
    x_ptr = x.contiguous().data_ptr()
    y_ptr = y.contiguous().data_ptr()
    out_ptr = out.contiguous().data_ptr()

    # Launch the kernel
    launch_kernel(_swiglu_fwd_kernel, grid, (x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE))

# Example usage
if __name__ == "__main__":
    N = 1024 * 1024
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    out = torch.empty_like(x)

    _swiglu_fwd(x, y, out)

    print(out)
