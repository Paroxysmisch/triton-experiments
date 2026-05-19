import triton
import triton.language as tl
import torch

# Kernel function to compute radians to degrees and square root
@triton.jit
def rad2deg_sqrt_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute radians to degrees
    b_value = a_value * (180.0 / tl.math.pi)
    # Compute the square root of the loaded elements
    c_value = tl.sqrt(a_value.to(tl.float32))
    # Store the results in output tensors 'b' and 'c' with boundary mask
    tl.store(b + offset, b_value, mask=mask)
    tl.store(c + offset, c_value, mask=mask)

# Wrapper function to launch the Triton kernel
def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create output tensors 'B' and 'C' with the same shape as 'input'
    B = torch.empty_like(input)
    C = torch.empty_like(input)
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    rad2deg_sqrt_kernel[(grid_size, 1, 1)](input, B, C, n_elements, block_size)
    return B, C
