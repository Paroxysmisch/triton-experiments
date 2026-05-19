import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def floor_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current block's index
    pid = tl.program_id(axis=0)
    # Calculate the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = offsets < n_elements

    # Load the elements from the input tensor
    input_elements = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Apply the floor operation
    output_elements = tl.floor(input_elements)
    # Store the results in the output tensor
    tl.store(output_ptr + offsets, output_elements, mask=mask)

def floor(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Check if the input tensor is on the GPU
    assert input.is_cuda, 'Input tensor must be on GPU'
    
    # For integer inputs, return a copy of the input tensor
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        return input.clone()

    # Determine the number of elements in the input tensor
    n_elements = input.numel()
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.is_cuda, 'Output tensor must be on GPU'
        assert out.shape == input.shape, 'Output tensor shape must match input tensor shape'
    
    # Determine the block size for parallel execution
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    # Calculate the grid size (number of blocks)
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the Triton kernel
    floor_kernel[(grid_size,)](
        input_ptr=input, output_ptr=out, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
