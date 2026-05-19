import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def floor_kernel(
    input_ptr, 
    output_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the global index for the current element
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = idx < N

    # Load the elements from input_ptr with masking
    input_elements = tl.load(input_ptr + idx, mask=mask)
    
    # Compute the floor of the elements
    floored_elements = tl.floor(input_elements)
    
    # Store the result in output_ptr with masking
    tl.store(output_ptr + idx, floored_elements, mask=mask)

def floor(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Determine the number of elements in the input tensor
    N = input.numel()
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and out.is_cuda, 'Both input and output tensors must be on GPU'
    
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Launch the Triton kernel
    floor_kernel[(N // BLOCK_SIZE + 1,)](
        input_ptr=input, 
        output_ptr=out, 
        N=N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
