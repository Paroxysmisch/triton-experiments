import torch
import triton
import triton.language as tl

@triton.jit
def puzzle1_kernel(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    value,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + value
    tl.store(output_ptr + offsets, output, mask=mask)

def puzzle1(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    N = x.numel()

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    return output

x = torch.randn(1024, device="cuda")
result = puzzle1(x)
print(result)
