import torch
import triton
import triton.language as tl
import scipy.special

# Triton kernel for Bessel function of the first kind of order 1
@triton.jit
def bessel_j1_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define a program index for each block
    pid = tl.program_id(axis=0)
    
    # Create a block of indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle the last block if it is smaller than BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute the Bessel function of the first kind of order 1
    y = scipy.special.j1(x)  # Using SciPy for Bessel function computation
    
    # Store the result
    tl.store(output_ptr + offsets, y, mask=mask)

# Wrapper function
def bessel_j1(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if not isinstance(out, torch.Tensor):
            raise ValueError("Output must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as the input tensor")

    # Define block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    bessel_j1_kernel[grid_size](input, out, input.numel(), BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
input_tensor = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
output_tensor = bessel_j1(input_tensor)
print(output_tensor)
