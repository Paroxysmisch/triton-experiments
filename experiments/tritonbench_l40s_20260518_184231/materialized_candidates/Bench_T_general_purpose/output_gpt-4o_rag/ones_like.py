import torch
import triton
import triton.language as tl

# Triton kernel that sets all elements in the output tensor to 1.0
@triton.jit
def ones_like_kernel(
    output_ptr,  # Pointer to the output tensor in GPU memory
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Size of each block of threads
):
    pid = tl.program_id(axis=0)  # Get the block index
    block_start = pid * BLOCK_SIZE  # Calculate the start index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Calculate offsets for each thread
    mask = offsets < n_elements  # Mask to ensure we don't write out of bounds
    tl.store(output_ptr + offsets, 1.0, mask=mask)  # Store 1.0 in all valid positions

# Function to initialize a tensor of the same size as the input tensor with ones using the Triton kernel
def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    if dtype is None:
        dtype = input.dtype  # Default to the dtype of the input tensor
    if device is None:
        device = input.device  # Default to the device of the input tensor

    out = torch.empty_like(input, dtype=dtype, device=device, memory_format=memory_format)  # Create an empty tensor
    N = input.numel()  # Calculate the total number of elements
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)  # Determine the grid size
    with torch.cuda.device(device):
        ones_like_kernel[grid_fn](out, N, BLOCK_SIZE=1024)  # Launch the Triton kernel
    out.requires_grad = requires_grad  # Set the requires_grad attribute
    return out  # Return the initialized tensor
