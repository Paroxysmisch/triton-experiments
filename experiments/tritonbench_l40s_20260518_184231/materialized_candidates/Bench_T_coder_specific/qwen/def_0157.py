import torch
from typing import Tuple

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.device.type == 'cuda' and other.device.type == 'cuda', "Input tensors must be on CUDA device"
    
    # Determine the block size
    block_size = 256
    
    # Allocate output tensors
    signbit_output = torch.zeros_like(input, dtype=torch.bool)
    bitwise_and_output = torch.zeros_like(input, dtype=input.dtype)
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Launch the Triton kernel
    signbit_bitwise_and_kernel[grid_size=(n_elements // block_size + 1,), block_size=(block_size,)](
        input.data_ptr(),
        other.data_ptr(),
        signbit_output.data_ptr(),
        bitwise_and_output.data_ptr(),
        n_elements
    )
    
    return signbit_output, bitwise_and_output
