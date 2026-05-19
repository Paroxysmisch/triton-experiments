import torch
import triton
import triton.language as tl

@triton.jit
def trunc_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor, n_elements: tl.tensor,
):
    pid = tl.program_id(axis=0)
    block_start = pid * n_elements
    offsets = block_start + tl.arange(0, n_elements)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = tl.trunc(input_vals)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def trunc(input, *, out=None):
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, but got {type(input)}")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError(f"out must be a torch.Tensor, but got {type(out)}")
    if out is None:
        out = torch.empty_like(input, dtype=torch.int64)
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        n_elements = input.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["n_elements"]),)
        trunc_kernel[grid](input, out, n_elements)
    else:
        out = input.clone()
    return out
