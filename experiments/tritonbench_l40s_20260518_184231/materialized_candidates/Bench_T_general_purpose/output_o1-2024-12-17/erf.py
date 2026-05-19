import triton
import triton.language as tl

# Polynomial coefficients for approximate erf on [0,∞). For x < 0, erf(x) = -erf(-x).
# Approximation from:
# Abramowitz and Stegun, Eq. 7.1.26
# erf(x) ≈ 1 - (1 / (1 + p*x)) * exp(-x^2 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)
# with t = 1 / (1 + p*x). Here we adapt a simpler polynomial expansion.

@triton.jit
def _erf_kernel(
    in_ptr, out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    x = tl.load(in_ptr + idx, mask=mask)

    # Compute erf(x) for each element via a numerical approximation
    # Use the polynomial approximation from:
    # erf(x) ≈ sign * (1 - e^(-x^2) * poly(1/(1+p*x))) for x >= 0 (sign = 1) else sign = -1
    # Coefficients
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = 1.0 - 2.0 * tl.where(x < 0, 1.0, 0.0)
    ax = tl.abs(x)

    t = 1.0 / (1.0 + p * ax)
    exp_term = tl.exp(-ax * ax)
    poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1
    y = 1.0 - poly * exp_term
    y = sign * y

    tl.store(out_ptr + idx, y, mask=mask)


def erf(input, *, out=None):
    if out is None:
        import torch
        out = torch.empty_like(input)

    n = input.numel()
    # Ensure data is contiguous
    inp_contig = input.contiguous()
    out_contig = out.contiguous()

    BLOCK_SIZE = 1024
    grid = ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _erf_kernel[grid](
        inp_contig.data_ptr(),
        out_contig.data_ptr(),
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )

    if out is not out_contig:
        out.copy_(out_contig)
    return out
