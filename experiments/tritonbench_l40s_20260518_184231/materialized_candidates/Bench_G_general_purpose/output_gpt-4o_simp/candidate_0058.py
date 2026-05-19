import triton
import triton.language as tl
import torch

@triton.jit
def isfinite_func_kernel_rank_1(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program index
    pid = tl.program_id(0)
    
    # Compute the block start and end indices
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    # Clamp the block end to the number of elements
    block_end = tl.min(block_end, n_elements)
    
    # Iterate over the block of elements
    for i in range(block_start, block_end):
        # Load the input element
        input_element = tl.load(input_ptr + i)
        
        # Check if the element is finite
        is_finite = tl.isfinite(input_element)
        
        # Store the result
        tl.store(output_ptr + i, is_finite)

def isfinite_func_wrapper_rank_1(input_tensor, output_tensor):
    # Ensure the input and output tensors are on the GPU
    assert input_tensor.is_cuda and output_tensor.is_cuda
    
    # Get the number of elements in the input tensor
    n_elements = input_tensor.numel()
    
    # Define the block size
    BLOCK_SIZE = 1024
    
    # Calculate the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    isfinite_func_kernel_rank_1[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
input_tensor = torch.tensor([1.0, float('inf'), 3.0, float('nan')], dtype=torch.float32, device='cuda')
output_tensor = torch.empty_like(input_tensor, dtype=torch.bool, device='cuda')

isfinite_func_wrapper_rank_1(input_tensor, output_tensor)

print(output_tensor.cpu())  # Should print tensor([ True, False,  True, False])
