import triton
import triton.language as tl
import torch

# Triton kernel to check if each element in a tensor is finite
@triton.jit
def isfinite_func_kernel_rank_1(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    # Compute the block ID
    pid = tl.program_id(axis=0)
    # Compute the start and end indices for the block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load the input data
    input_block = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    # Apply the isfinite function
    output_block = tl.libdevice.isfinite(input_block)
    # Store the results
    tl.store(output_ptr + offsets, output_block, mask=offsets < n_elements)

def isfinite_func_wrapper_rank_1(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    # Ensure the input and output tensors are on the same device
    assert input_tensor.device == output_tensor.device, "Input and output tensors must be on the same device"
    # Ensure the input and output tensors have the same shape
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    # Get the number of elements in the tensor
    n_elements = input_tensor.numel()
    
    # Determine the optimal block size
    BLOCK_SIZE = 1024  # Adjust this based on your GPU architecture and problem size
    
    # Determine the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    isfinite_func_kernel_rank_1[grid, BLOCK_SIZE](
        input_tensor,  # Pointer to the input tensor
        output_tensor,  # Pointer to the output tensor
        n_elements,  # Number of elements in the tensor
        BLOCK_SIZE  # Block size for parallel processing
    )

# Create a sample input tensor
input_tensor = torch.tensor([1.0, 2.0, float('inf'), float('nan'), 4.0], device='cuda')

# Create an output tensor of the same shape with boolean type
output_tensor = torch.empty_like(input_tensor, dtype=torch.bool, device='cuda')

# Call the wrapper function
isfinite_func_wrapper_rank_1(input_tensor, output_tensor)

# Print the results
print(output_tensor)
