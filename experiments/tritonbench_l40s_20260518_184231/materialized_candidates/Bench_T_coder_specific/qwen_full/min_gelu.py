import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from torch._inductor.triton_heuristics import reduction
from torch._inductor.codecache import AsyncCompile

_gaussian_pdf_normalization = 1 / (math.sqrt(2 * math.pi))

@triton.jit
def _triton_gaussian_cdf(x):
    # Using approximation introduces greater parity errors.
    # return tl.sigmoid(1.702 * x) * 0.5
    return tl.math.erf(x / 1.41421356237) * 0.5 + 0.5

@triton.jit
def _triton_gaussian_pdf(x):
    return (tl.exp(-0.5 * x * x) * _gaussian_pdf_normalization)

@triton.jit
def _triton_gaussian_gelu(x, approximate: tl.constexpr):
    if approximate == "tanh":
        # Using tanh introduces greater parity errors.
        # return 0.5 * x * (1 + _triton_gaussian_cdf(x))
        return 0.5 * x * (1 + tl.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    else:
        return x * _triton_gaussian_cdf(x)

@triton.jit
def min_gelu_kernel(
    inp,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp + offsets, mask=mask)
    x = _triton_gaussian_gelu(x, approximate="tanh")
    tl.store(out + offsets, x, mask=mask)

 AsyncCompile(
        min_gelu_kernel,
        signature={0: "*fp32", 1: "*fp32", 2: "i64"},
        device=0,
        constants={3: 1024},
    ).wait()

def min_gelu(
    input: Tensor,
    dim: Optional[int] = None,
    keepdim: Optional[bool] = False,
    approximate: Optional[str] = "none",
    out: Optional[Tensor] = None,
) -> Tensor:
    if dim is None or approximate == "tanh":
        if approximate == "tanh":
            input = input.float()
        output = torch.min(gelu(input, approximate=approximate))
        if not keepdim:
            output = output.squeeze(dim)
        return output

    if out is None:
        out = torch.empty_like(input, dtype=input.dtype)
    else:
        assert out.shape == input.shape

    assert input.is_contiguous()
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    min_gelu_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
