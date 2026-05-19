import torch
import triton
import triton.language as tl

@triton.jit
def relu_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    relu_vals = tl.maximum(input_vals, 0)
    sqrt_vals = tl.sqrt(relu_vals)
    tl.store(output_ptr + offsets, sqrt_vals, mask=mask)

def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    if out is None and inplace:
        if not input.is_contiguous():
            raise RuntimeError("Cannot perform inplace operation on non-contiguous tensor")
        output = input
    else:
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out
            if output.shape != input.shape:
                raise RuntimeError(f"Output shape {output.shape} does not match input shape {input.shape}")
            if output.dtype != input.dtype:
                raise RuntimeError(f"Output dtype {output.dtype} does not match input dtype {input.dtype}")
        if not output.is_contiguous():
            output = output.contiguous()
    
    if not input.is_contiguous():
        input = input.contiguous()
    
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']), )
    relu_sqrt_kernel[grid](input, output, num_elements, BLOCK_SIZE=1024)
    
    return output
