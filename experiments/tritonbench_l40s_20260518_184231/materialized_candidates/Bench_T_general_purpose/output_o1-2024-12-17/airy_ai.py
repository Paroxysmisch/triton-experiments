import triton
import triton.language as tl
import torch

# Triton kernel to compute an approximate Airy Ai for each element.
# This uses a simple polynomial approximation for demonstration.
# Note: For production usage, consider a more accurate piecewise or library-based approach.
@triton.jit
def _airy_ai_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets, mask=mask)

    # Polynomial approximation around 0 for Ai(x). 
    # This is only a rough approximation for demonstration purposes.
    # Ai(x) ~ a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + ...
    a0 = 0.3550280538878172
    a1 = -0.2588194037928068
    a2 = 0.02026349177571668
    a3 = -0.00110847623689453
    a4 = 3.993369402954306e-05
    a5 = -1.1313099550900505e-06
    a6 = 2.3929247998650912e-08

    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3

    result = a0 + a1*x + a2*x2 + a3*x3 + a4*x4 + a5*x5 + a6*x6

    tl.store(out_ptr + offsets, result, mask=mask)


def airy_ai(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure input is a CUDA tensor
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor.")

    n = input.numel()
    if out is None:
        out = torch.empty_like(input)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _airy_ai_kernel[grid](input.data_ptr(),
                          out.data_ptr(),
                          n,
                          BLOCK_SIZE=BLOCK_SIZE)

    return out
