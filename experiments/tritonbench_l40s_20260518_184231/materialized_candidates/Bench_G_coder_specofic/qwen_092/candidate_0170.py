import triton
import triton.language as tl
import torch

@triton.jit
def cos_func(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the global index of the current thread
    pid = tl.program_id(axis=0)
    # Calculate the starting index for the current block
    offset = pid * BLOCK_SIZE
    # Calculate the number of elements each block will process
    num_elements = min(BLOCK_SIZE, n_elements - offset)
    
    # Define a range of indices for the current block
    offsets = offset + tl.arange(0, num_elements)
    # Load elements from the input tensor 'a'
    a_value = tl.load(a_ptr + offsets, mask=offsets < n_elements)
    
    # Compute the cosine of each element
    b_value = tl.cos(a_value)
    
    # Store the results in the output tensor 'b'
    tl.store(b_ptr + offsets, b_value, mask=offsets < n_elements)

def cos(a, BLOCK_SIZE=None):
    if BLOCK_SIZE is None:
        # Calculate the nearest power of 2 greater than the square root of the number of elements
        n_elements = a.numel()
        BLOCK_SIZE = 2 ** (int(n_elements.bit_length() / 2) + 1)
    
    # Allocate memory for the output tensor
    b = torch.empty_like(a)
    
    # Get the grid size
    grid_size = (a.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    cos_func[grid_size, BLOCK_SIZE](a, b, a.numel(), BLOCK_SIZE)
    
    return b
