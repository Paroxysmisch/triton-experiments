import torch
import triton
import triton.language as tl
from typing import Optional, Union

def rand(
    *size,
    generator: Optional[torch.Generator] = None,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = torch.strided,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: Optional[bool] = False,
    pin_memory: Optional[bool] = False,
) -> torch.Tensor:
    """Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
        generator (torch.Generator, optional): a pseudorandom number generator for sampling.
        out (Tensor, optional): the output tensor.
        dtype (torch.dtype, optional): the desired data type of returned tensor.
        layout (torch.layout, optional): the desired layout of returned Tensor.
        device (torch.device, optional): the desired device of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
        pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory.

    Returns:
        Tensor: A tensor filled with random numbers.
    """
    if out is None:
        out = torch.empty(*size, dtype=dtype, device=device, pin_memory=pin_memory)
    elif out.shape != size:
        raise ValueError("shape of out and size must be the same")

    # Call the Triton kernel to fill the tensor with random numbers
    _rand_triton[(out.shape)](out)
    return out

@triton.jit
def _rand_triton(out_ptr: torch.Tensor):
    """Kernel to generate random numbers in [0, 1) for each element in the output tensor."""
    # Get the index of the current element
    idx = tl.program_id(0)
    
    # Generate a random number
    random_number = tl.rand()
    
    # Store the random number in the output tensor
    tl.store(out_ptr + idx, random_number)
