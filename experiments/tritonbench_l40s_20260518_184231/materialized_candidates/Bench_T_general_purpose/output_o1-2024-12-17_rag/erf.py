import torch
import triton
import triton.language as tl

# Polynomial approximation constants for erf, based on Abramowitz and Stegun formula 7.1.26
_a1 = 0.254829592
_a2 = -0.284496736
_a3 = 1.421413741
_a4 = -1.453152027
_a5 = 1.061405429
_p = 0.3275911

@triton.jit
def erf_kernel(
    A_ptr,       # pointer to input
    O_ptr,       # pointer to output
    N,           # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    sign_x = tl.where(x < 0.0, -1.0, 1.0)
    x_abs = tl.abs(x)

    t = 1.0 / (1.0 + _p * x_abs)
    # Polynomial approximation
    poly = (_a1 * t +
            _a2 * (t * t) +
            _a3 * (t * t * t) +
            _a4 * (t * t * t * t) +
            _a5 * (t * t * t * t * t))
    erf_approx = 1.0 - poly * tl.exp(-x_abs * x_abs)
    val = sign_x * erf_approx

    tl.store(O_ptr + offsets, val, mask=mask)

def erf(input: torch.Tensor, *, out=None) -> torch.Tensor:
    # Ensure input is on CUDA
    assert input.is_cuda, "Input tensor must be on CUDA."
    if out is not None:
        assert out.is_cuda, "Output tensor must be on CUDA if provided."
        assert out.shape == input.shape, "Output tensor must have the same shape as input."
    else:
        out = torch.empty_like(input)

    N = input.numel()
    BLOCK_SIZE = 1024
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    erf_kernel[grid](
        A_ptr=input,
        O_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
