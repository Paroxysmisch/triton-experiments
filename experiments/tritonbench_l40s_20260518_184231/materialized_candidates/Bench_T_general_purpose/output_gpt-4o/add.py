import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    input_ptr, other_ptr, out_ptr, alpha,
    input_stride, other_stride, out_stride,
    input_size, other_size, out_size,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block index
    block_idx = tl.program_id(0)
    # Compute the starting index for this block
    start_idx = block_idx * BLOCK_SIZE
    # Create a range of indices for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Load data from the input and other tensors
    input_val = tl.load(input_ptr + offsets * input_stride, mask=offsets < input_size, other=0)
    other_val = tl.load(other_ptr + offsets * other_stride, mask=offsets < other_size, other=0)
    
    # Perform the addition with scaling
    result = input_val + alpha * other_val
    
    # Store the result
    tl.store(out_ptr + offsets * out_stride, result, mask=offsets < out_size)

def add(input, other, *, alpha=1, out=None):
    # Convert 'other' to a tensor if it's a scalar
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast input and other to a common shape
    input, other = torch.broadcast_tensors(input, other)
    
    # If out is not provided, create a new tensor for the output
    if out is None:
        out = torch.empty_like(input)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(input.numel(), meta['BLOCK_SIZE']),)
    add_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        out_ptr=out,
        alpha=alpha,
        input_stride=input.stride(0),
        other_stride=other.stride(0),
        out_stride=out.stride(0),
        input_size=input.numel(),
        other_size=other.numel(),
        out_size=out.numel(),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
