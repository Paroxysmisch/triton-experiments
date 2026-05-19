import triton
import triton.language as tl

@triton.jit
def logspace_kernel(start, end, steps, base, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Compute the linear space in the exponent domain
    step_size = (end - start) / (steps - 1)
    exponents = start + offset * step_size
    
    # Calculate the logspace values
    logspace_values = base ** exponents
    
    # Store the results
    tl.store(output_ptr + offset, logspace_values)

import torch

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Handle scalar tensor inputs
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    # Set default dtype and device if not provided
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(steps, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)

    # Ensure the output tensor has the correct size
    assert out.size(0) == steps, "Output tensor has incorrect size"

    # Define block size
    BLOCK_SIZE = 128  # Example block size, adjust based on hardware

    # Launch the Triton kernel
    grid = (steps + BLOCK_SIZE - 1) // BLOCK_SIZE
    logspace_kernel[grid](start, end, steps, base, out, BLOCK_SIZE=BLOCK_SIZE)

    return out
