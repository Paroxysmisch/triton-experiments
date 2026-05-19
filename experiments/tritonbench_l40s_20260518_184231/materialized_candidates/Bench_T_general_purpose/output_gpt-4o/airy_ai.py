import triton
import triton.language as tl
import torch
import scipy.special

@triton.jit
def airy_ai_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the program ID and block size
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from input tensor
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the Airy Ai function for each element
    result = tl.libdevice.airy_ai(input_data)
    
    # Store the result in the output tensor
    tl.store(output_ptr + offsets, result, mask=mask)

def airy_ai(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Determine the number of elements in the input tensor
    n_elements = input.numel()
    
    # If no output tensor is provided, create one
    if out is None:
        out = torch.empty_like(input)
    elif not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor if provided")
    
    # Ensure the output tensor is the correct size
    assert out.numel() == n_elements, "out tensor must have the same number of elements as input"
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Define an appropriate block size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    airy_ai_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
