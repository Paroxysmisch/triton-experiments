import triton
import triton.language as tl

@triton.jit
def logspace_kernel(start, end, steps, base, out_ptr, n_elements):
    idx = tl.program_id(0)
    if idx < n_elements:
        # Calculate the logarithmic progression
        step_size = (end - start) / (steps - 1)
        exponent = start + idx * step_size
        out_ptr[idx] = base ** exponent

import torch

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
    # Validate inputs
    if isinstance(start, torch.Tensor) and start.dim() != 0:
        raise ValueError("start must be a 0-dimensional tensor or a float")
    if isinstance(end, torch.Tensor) and end.dim() != 0:
        raise ValueError("end must be a 0-dimensional tensor or a float")
    
    # Prepare output tensor
    if out is None:
        out = torch.empty(steps, dtype=dtype, layout=layout, device=device)
    
    # Get the number of elements
    n_elements = out.numel()
    
    # Launch the Triton kernel
    logspace_kernel[(n_elements,)](start, end, steps, base, out_ptr=out.data_ptr(), n_elements=n_elements)
    
    return out
