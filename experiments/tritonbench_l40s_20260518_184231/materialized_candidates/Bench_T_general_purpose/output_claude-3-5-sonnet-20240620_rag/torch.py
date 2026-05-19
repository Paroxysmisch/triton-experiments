import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def permute_kernel(
        input_ptr,
        output_ptr,
        dims_ptr,
        N,  # Total number of elements in the input tensor
        D,  # Number of dimensions
        BLOCK_SIZE: tl.constexpr
):
    # Get the global index of the current thread
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = idx < N

    # Load the dimensions from dims_ptr
    dims = tl.load(dims_ptr)

    # Calculate the new indices based on the permutation
    new_indices = tl.zeros((D,), dtype=tl.int32)
    for i in range(D):
        new_indices[i] = tl.load(dims + i)

    # Calculate the original indices based on the new indices
    original_indices = tl.zeros((D,), dtype=tl.int32)
    for i in range(D):
        original_indices[i] = idx // (N // (D - i)) % (N // (D - i + 1))

    # Load the input tensor elements using the original indices
    input_elements = tl.load(input_ptr + tl.reduce(tl.arange(D), lambda acc, i: acc * original_indices[i], initial=0), mask=mask)

    # Store the result in the output tensor
    tl.store(output_ptr + idx, input_elements, mask=mask)

def permute_copy(input: torch.Tensor, dims: list) -> torch.Tensor:
    # Get the total number of elements and dimensions of the input tensor
    N = input.numel()
    D = input.dim()
    
    # Prepare an output tensor on the same device
    output = torch.empty(size=input.size()).to(device)

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and output.is_cuda, 'One of the tensors is not on GPU'

    # Prepare the dims tensor
    dims_tensor = torch.tensor(dims, dtype=torch.int32).to(device)

    # Block size can be set to a reasonable value
    BLOCK_SIZE = 256  # You can adjust this based on your needs
    print(f'Block size: {BLOCK_SIZE}, grid: {(N // BLOCK_SIZE + 1,)}')

    # Launch the Triton kernel
    permute_kernel[(N // BLOCK_SIZE + 1,)](
        input_ptr=input, 
        output_ptr=output, 
        dims_ptr=dims_tensor, 
        N=N, 
        D=D, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
