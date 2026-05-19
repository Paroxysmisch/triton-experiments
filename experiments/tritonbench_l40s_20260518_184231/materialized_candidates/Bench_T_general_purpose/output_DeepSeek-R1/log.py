import torch
import triton
import triton.language as tl

@triton.jit
def _log_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.log(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def log(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input)}")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"out must be a torch.Tensor, got {type(out)}")
        if out.shape != input.shape:
            raise ValueError(f"out shape {out.shape} does not match input shape {input.shape}")
        if out.dtype != input.dtype:
            raise TypeError(f"out dtype {out.dtype} does not match input dtype {input.dtype}")
    else:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _log_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )
    return out
