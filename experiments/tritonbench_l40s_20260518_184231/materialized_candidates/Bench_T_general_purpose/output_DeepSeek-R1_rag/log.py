import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate input and output offsets using strides
    input_offsets = offsets * input_stride
    output_offsets = offsets * output_stride
    
    # Load input elements
    x = tl.load(input_ptr + input_offsets, mask=mask)
    
    # Compute natural logarithm
    y = tl.log(x)
    
    # Store results
    tl.store(output_ptr + output_offsets, y, mask=mask)

def log(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure output tensor is properly shaped
    if out is not None and out.shape != input.shape:
        raise ValueError("Output tensor must have the same shape as input")
    if out is None:
        out = torch.empty_like(input)
    
    # Flatten tensors to 1D view for processing
    input_1d = input.view(-1)
    output_1d = out.view(-1)
    n_elements = input_1d.numel()
    
    # Set block size and compute grid dimensions
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    log_kernel[grid](
        output_1d.data_ptr(),
        input_1d.data_ptr(),
        n_elements,
        input_1d.stride(0),
        output_1d.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
