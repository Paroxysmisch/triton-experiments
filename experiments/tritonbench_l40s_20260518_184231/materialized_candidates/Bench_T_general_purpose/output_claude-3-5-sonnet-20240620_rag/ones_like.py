import torch
import triton
import triton.language as tl

@triton.jit
def ones_like_kernel(
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of thread block
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Store ones in the output tensor
    tl.store(output_ptr + offsets, 1.0, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, 
              requires_grad=False, memory_format=torch.preserve_format) -> torch.Tensor:
    # Use input tensor's properties if not specified
    dtype = dtype if dtype is not None else input.dtype
    device = device if device is not None else input.device
    layout = layout if layout is not None else input.layout
    
    # Create empty tensor with same shape as input
    out = torch.empty_like(
        input,
        dtype=dtype,
        device=device,
        layout=layout,
        memory_format=memory_format
    )
    
    # Only run Triton kernel if on CUDA device
    if device.type == 'cuda':
        n_elements = input.numel()
        BLOCK_SIZE = 1024
        
        # Calculate grid size
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Launch kernel
        ones_like_kernel[grid](
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fallback to CPU implementation
        out.fill_(1)
    
    # Set requires_grad if specified
    out.requires_grad_(requires_grad)
    
    return out
