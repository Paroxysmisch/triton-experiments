import torch
import triton
import triton.language as tl

@triton.jit
def reciprocal_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    output_values = 1.0 / input_values
    tl.store(output_ptr + offsets, output_values, mask=mask)

def reciprocal(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Promote integral inputs to the default scalar type
    if not input.dtype.is_floating_point:
        promoted_input = input.to(torch.get_default_dtype())
    else:
        promoted_input = input
    
    # Validate or create the output tensor
    if out is None:
        out = torch.empty_like(promoted_input)
    else:
        if out.shape != input.shape:
            raise RuntimeError("out shape must match input shape")
        if out.dtype != promoted_input.dtype:
            raise RuntimeError(f"out.dtype must be {promoted_input.dtype}, but got {out.dtype}")
    
    # Ensure contiguous tensors for kernel efficiency
    promoted_input = promoted_input.contiguous()
    out = out.contiguous()
    
    n_elements = promoted_input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    reciprocal_kernel[grid](
        promoted_input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return out
