import torch
import triton
import triton.language as tl

@triton.jit
def polygamma_kernel(
    input_ptr,
    output_ptr,
    n,
    input_numel,
    MAX_TERMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_numel

    x = tl.load(input_ptr + offsets, mask=mask)

    # Calculate sign (-1)^(n+1)
    sign = (-1.0) ** (n + 1)
    # Calculate n!
    factorial_n = 1.0
    for i in range(1, n + 1):
        factorial_n *= i

    # Sum terms from k=0 to MAX_TERMS-1
    sum_ = 0.0
    for k in range(MAX_TERMS):
        term = 1.0 / tl.pow(x + k, n + 1)
        sum_ += term

    result = sign * factorial_n * sum_
    tl.store(output_ptr + offsets, result, mask=mask)

def polygamma(n: int, input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if n < 0:
        raise ValueError("polygamma(n, input) requires n >= 0")
    if input.is_cuda:
        if out is None:
            out = torch.empty_like(input)
        assert out.is_cuda, "Output tensor must be on CUDA if input is on CUDA"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        input_numel = input.numel()
        if input_numel == 0:
            return out
        MAX_TERMS = 1024  # Adjust based on desired accuracy
        BLOCK_SIZE = 128
        grid = lambda meta: (triton.cdiv(input_numel, meta['BLOCK_SIZE']),)
        polygamma_kernel[grid](
            input, out, n, input_numel,
            MAX_TERMS=MAX_TERMS, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        if out is not None:
            return torch.polygamma(n, input, out=out)
        else:
            return torch.polygamma(n, input)
