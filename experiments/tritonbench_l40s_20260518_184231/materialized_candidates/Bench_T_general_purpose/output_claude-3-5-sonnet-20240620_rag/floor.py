import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def floor_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the index for the current program
    index = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = index < N

    # Load the input elements
    input_values = tl.load(input_ptr + index, mask=mask)

    # Compute the floor of the input values
    output_values = tl.floor(input_values)

    # Store the result in output_ptr
    tl.store(output_ptr + index, output_values, mask=mask)

def floor(input: torch.Tensor, *, out=None) -> torch.Tensor:
    # Check if the input is an integer tensor
    if input.dtype in [torch.int32, torch.int64]:
        return input.clone()  # Return a copy for integer inputs

    # Get the number of elements in the input tensor
    N = input.numel()
    # Prepare an output tensor on the same device
    output = torch.empty_like(input) if out is None else out

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and output.is_cuda, 'One of the tensors is not on GPU'

    # Block size can be set to a suitable value
    BLOCK_SIZE = 1024  # You can adjust this based on your needs
    print(f'Block size: {BLOCK_SIZE}, grid: {(N + BLOCK_SIZE - 1) // BLOCK_SIZE}')

    # Launch the Triton kernel
    floor_kernel[(N + BLOCK_SIZE - 1) // BLOCK_SIZE](input_ptr=input, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE)

    return output
