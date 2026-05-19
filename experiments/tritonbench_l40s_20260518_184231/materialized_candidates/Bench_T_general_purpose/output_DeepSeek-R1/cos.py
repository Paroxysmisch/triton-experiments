import torch
import triton
import triton.language as tl

@triton.jit
def cos_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.cos(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def cos(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "output shape must match input"
        assert out.dtype == input.dtype, "output dtype must match input"
    
    # Check if input is contiguous; if not, create a contiguous copy
    if not input.is_contiguous():
        input = input.contiguous()
    
    # Process output tensor to handle non-contiguity
    original_out = out
    if not out.is_contiguous():
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
    
    # Flatten to 1D for kernel processing
    input_flat = input.view(-1)
    out_flat = out.view(-1)
    n_elements = input_flat.numel()
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    cos_kernel[grid](input_flat, out_flat, n_elements, BLOCK_SIZE=1024)
    
    # If original output was non-contiguous, copy results back
    if original_out is not out:
        original_out.copy_(out)
    
    return original_out
