import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def tanh_kernel(
    input_ptr, 
    output_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the index for the current program
    index = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = index < N
    
    # Load the input elements
    input_elements = tl.load(input_ptr + index, mask=mask)
    # Compute the hyperbolic tangent
    tanh_elements = tl.tanh(input_elements)
    # Store the result
    tl.store(output_ptr + index, tanh_elements, mask=mask)

def tanh(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Get the number of elements in the input tensor
    N = input.numel()
    # Prepare an output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and out.is_cuda, 'Both tensors must be on GPU'
    
    # Determine the block size for parallel execution
    BLOCK_SIZE = 1024  # You can choose an appropriate block size
    # Calculate the grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    tanh_kernel[grid](
        input_ptr=input, 
        output_ptr=out, 
        N=N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
