import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel that sets all elements in the output tensor to 1.0
@triton.jit
def ones_kernel(
    output_ptr,  # Pointer to the output tensor in GPU memory
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Size of each block of threads
):
    pid = tl.program_id(axis=0)  # Get the block index
    block_start = pid * BLOCK_SIZE  # Calculate the start index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Calculate offsets for each thread
    mask = offsets < n_elements  # Mask to ensure we don't write out of bounds
    tl.store(output_ptr + offsets, 1.0, mask=mask)  # Store 1.0 in all valid positions

# Function to initialize a tensor of given size with ones using the Triton kernel
def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        dtype = torch.get_default_dtype()  # Use default PyTorch dtype if none provided
    if device is None:
        device = torch.device("cuda")  # Default to CUDA device

    out = torch.empty(size, device=device, dtype=dtype)  # Create an empty tensor
    N = volume(size)  # Calculate the total number of elements
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)  # Determine the grid size
    with torch.cuda.device(device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=1024)  # Launch the Triton kernel
    return out  # Return the initialized tensor
