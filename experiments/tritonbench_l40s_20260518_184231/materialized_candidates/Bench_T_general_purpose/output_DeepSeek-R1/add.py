import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets, mask=mask, other=0)
    other = tl.load(other_ptr + offsets, mask=mask, other=0)
    output = input + other
    tl.store(output_ptr + offsets, output, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Convert 'other' to a tensor if it's a number
    other_tensor = other if isinstance(other, torch.Tensor) else torch.tensor(other, device=input.device)
    
    # Scale 'other' by alpha
    scaled_other = other_tensor * alpha
    
    # Broadcast input and scaled_other to a common shape
    try:
        broadcasted_input, broadcasted_other = torch.broadcast_tensors(input, scaled_other)
    except RuntimeError as e:
        raise ValueError("Shapes of input and other are not broadcastable") from e
    
    # Determine the output dtype
    output_dtype = torch.result_type(broadcasted_input, broadcasted_other)
    
    # Ensure the output tensor is properly configured
    if out is None:
        out = torch.empty_like(broadcasted_input, dtype=output_dtype)
    else:
        if out.shape != broadcasted_input.shape:
            raise ValueError("Output tensor shape does not match broadcasted shape")
        if out.dtype != output_dtype:
            raise ValueError(f"Output tensor dtype {out.dtype} does not match expected {output_dtype}")
    
    # Cast to the correct dtype if necessary
    broadcasted_input = broadcasted_input.to(output_dtype)
    broadcasted_other = broadcasted_other.to(output_dtype)
    
    # Ensure contiguous tensors
    contiguous_input = broadcasted_input.contiguous()
    contiguous_other = broadcasted_other.contiguous()
    contiguous_out = out.contiguous()
    
    # Launch kernel
    n_elements = contiguous_out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](
        contiguous_input, contiguous_other, contiguous_out,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    # Copy back if out was non-contiguous
    if not out.is_contiguous():
        out.copy_(contiguous_out)
    
    return out
