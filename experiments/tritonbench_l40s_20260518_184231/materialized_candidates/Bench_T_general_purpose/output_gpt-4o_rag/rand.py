import torch
import triton
import triton.language as tl
from typing import Optional, Union

def rand(
    *size,
    generator: Optional[torch.Generator] = None,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
    pin_memory: bool = False
) -> torch.Tensor:
    """Generates a tensor filled with random numbers from a uniform distribution on [0, 1)."""
    
    if out is None:
        out = torch.empty(*size, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    elif out.shape != size:
        raise ValueError("shape of out and size must be the same")
    
    # Set default dtype and device if not provided
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_elements = out.numel()
    block_size = 1024
    grid_size = (n_elements + block_size - 1) // block_size
    
    _rand_triton[grid_size](
        out, n_elements, generator=generator, block_size=block_size
    )
    
    return out

@triton.jit
def _rand_triton(
    out_ptr: tl.pointer,
    n_elements: tl.int32,
    generator: tl.pointer,
    block_size: tl.constexpr
):
    """Triton kernel to generate random numbers."""
    
    idx = tl.program_id(axis=0) * block_size + tl.arange(0, block_size)
    mask = idx < n_elements
    
    # Generate random numbers using Triton's random function
    random_values = tl.rand(generator, idx)
    
    # Store the generated random numbers in the output tensor
    tl.store(out_ptr + idx, random_values, mask=mask)
