import torch
import triton
import triton.language as tl

@triton.jit
def trunc_kernel(
    input_ptr,
    output_ptr,
    nelems,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nelems

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    truncated = tl.math.trunc(input_vals)
    tl.store(output_ptr + offsets, truncated, mask=mask)

def trunc(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                       torch.uint8, torch.bool):
        if out is not None:
            if out.shape != input.shape:
                raise ValueError("out tensor must have the same shape as input")
            out.copy_(input)
            return out
        else:
            return input.clone()
    else:
        input_contiguous = input.contiguous()
        if out is not None:
            if out.shape != input.shape:
                raise ValueError("out tensor must have the same shape as input")
            output = out if out.is_contiguous() else torch.empty_like(input_contiguous)
        else:
            output = torch.empty_like(input_contiguous)

        nelems = input_contiguous.numel()
        if nelems == 0:
            return output  # handle empty tensor

        grid = lambda meta: (triton.cdiv(nelems, meta['BLOCK_SIZE']),)
        trunc_kernel[grid](input_contiguous, output, nelems, BLOCK_SIZE=1024)

        if out is not None and not out.is_contiguous():
            out.copy_(output)
            return out
        return output
