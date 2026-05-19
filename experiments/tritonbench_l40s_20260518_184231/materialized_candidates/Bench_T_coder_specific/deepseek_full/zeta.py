import torch
import triton
import triton.language as tl

@triton.jit
def zeta_kernel(x, q, OUTPUT):
    # Triton kernel for computing the Hurwitz zeta function
    idx = tl.program_id(0)
    q_val = tl.load(q + idx)
    if q_val == 1:
        # Special case for the Riemann zeta function
        x_val = tl.load(x + idx)
        series_sum = tl.rsum(tl.exp2(x_val * tl.arange(0, 128)))
        result = -series_sum / (2 ** x_val)
    else:
        k = tl.arange(0, 128)
        series_sum = tl.sum(tl.rdivide(1, (k + q_val) ** x_val))
        result = -series_sum
    tl.store(OUTPUT + idx, result)

def zeta(x, q, *, out=None):
    # Function to call the Triton kernel
    assert x.is_contiguous()
    assert q.is_contiguous()
    assert x.is_floating_point()
    assert q.is_floating_point()
    assert x.shape == q.shape
    if out is None:
        out = torch.empty_like(q, dtype=torch.float32, device="cuda")
    else:
        assert out.shape == q.shape
        assert out.is_floating_point()
    assert x.is_cuda and q.is_cuda and out.is_cuda
    n_elements = out.numel()
    grid = lambda meta: (n_elements,)
    zeta_kernel[grid](x, q, out)
    return out
