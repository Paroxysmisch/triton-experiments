import triton
import triton.language as tl
import torch

# Triton kernel to compute the exponential of the elements in the input tensor
@triton.jit
def exp_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for the current block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we do not access out-of-bounds elements
    mask = offset < n_elements
    # Load the input elements
    a_value = tl.load(a + offset, mask=mask)
    # Compute the exponential of the loaded elements
    b_value = tl.exp(a_value.to(tl.float32))
    # Store the results in the output tensor
    tl.store(b + offset, b_value, mask=mask)

# Wrapper function to set up and launch the Triton kernel
def exp(input, *, out=None):
    # Determine the number of elements in the input tensor
    n_elements = input.numel()
    
    # If out is not provided, create a new tensor to store the result
    if out is None:
        out = torch.empty_like(input)
    
    # Calculate the block size based on the number of elements
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Determine the grid size required to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the Triton kernel 'exp_func' with the computed grid and block size
    exp_func[(grid_size, 1, 1)](input, out, n_elements, block_size)
    
    return out
