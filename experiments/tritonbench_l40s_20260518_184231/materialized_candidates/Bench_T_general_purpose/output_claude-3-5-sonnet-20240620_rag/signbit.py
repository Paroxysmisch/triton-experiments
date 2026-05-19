import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements):
    # Get the unique program ID for each element
    idx = tl.program_id(0)
    
    # Check if the index is within bounds
    if idx < n_elements:
        # Load the input value
        value = tl.load(input_ptr + idx)
        # Check if the sign bit is set (negative zero is handled)
        sign_bit = (value < 0) | (value == 0) & (tl.sign(value) < 0)
        # Store the result in the output tensor
        tl.store(output_ptr + idx, sign_bit)

def signbit(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    n_elements = input.numel()
    
    # Move input tensor to GPU if not already there
    input = input.cuda()
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    
    # Launch the kernel
    signbit_kernel[(n_elements,)](input, out, n_elements)
    
    return out
