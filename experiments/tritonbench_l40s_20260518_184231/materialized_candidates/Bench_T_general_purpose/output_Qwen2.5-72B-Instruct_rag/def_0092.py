import triton
import triton.language as tl
import torch

# Kernel function to compute the reciprocal of the square root of elements in a tensor
@triton.jit
def rsqrt_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the reciprocal of the square root of the loaded elements
    b_value = 1.0 / tl.sqrt(a_value.to(tl.float32))
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)

# Wrapper function to compute the reciprocal of the square root of the tensordot product
def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    # Compute the tensordot product of tensors 'a' and 'b'
    tensordot_result = torch.tensordot(a, b, dims=dims)
    
    # Create an output tensor 'B' with the same shape as the tensordot result
    B = torch.empty_like(tensordot_result)
    
    # Get the total number of elements in the tensordot result
    n_elements = tensordot_result.numel()
    
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the Triton kernel
    rsqrt_kernel[(grid_size, 1, 1)](tensordot_result, B, n_elements, block_size)
    
    return B
