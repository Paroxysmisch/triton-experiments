import triton
import triton.language as tl
import torch

@triton.jit
def _gammaln_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    x_abs = tl.abs(x)

    # Simple Stirling-based approximation for ln(Gamma(x))
    # Works best for x > 0
    def stirling_approx(v):
        return (v - 0.5) * tl.log(v) - v + 0.9189385332046727  # 0.5 * ln(2*pi)

    # Approximate ln(Gamma(|x|))
    # Handle |x| < 1 by shifting argument
    def approx_lngamma(v):
        # Protect against zero or very small input
        cond = v >= 1.0
        r = tl.where(cond, stirling_approx(v), stirling_approx(v + 1.0) - tl.log(v + 1e-20))
        return r

    out_val = approx_lngamma(x_abs)
    tl.store(out_ptr + offsets, out_val, mask=mask)

def gammaln(input, *, out=None):
    if not input.is_cuda:
        raise RuntimeError("gammaln: input must be a CUDA tensor")
    if out is not None and not out.is_cuda:
        raise RuntimeError("gammaln: out must be a CUDA tensor if provided")

    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    _gammaln_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
