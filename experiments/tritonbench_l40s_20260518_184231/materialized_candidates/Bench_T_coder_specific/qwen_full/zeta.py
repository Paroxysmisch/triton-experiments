import torch
import triton
import triton.language as tl

@triton.jit
def zeta_kernel(x, q, out, BLOCK_SIZE: tl.constexpr):
    # Compute the index of the element to process
    idx = tl.program_id(0)
    x_i = tl.load(x + idx)
    q_i = tl.load(q + idx)
    # Initialize the sum
    sum = 0.0
    # Compute the sum of the series
    k = tl.arange(0, BLOCK_SIZE)
    denom = tl.pow(k + q_i, x_i)
    sum += tl.sum(denom)
    # Store the result
    tl.store(out + idx, sum)

def zeta(input, other, *, out=None):
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_floating_point()
    # Set the block size for Triton kernel
    BLOCK_SIZE = 128
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(input.numel(), meta['BLOCK_SIZE']),)
    zeta_kernel[grid](input, other, out, BLOCK_SIZE=BLOCK_SIZE)
    return out
