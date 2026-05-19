import triton
import triton.language as tl
import torch

@triton.jit
def isfinite_func_kernel_rank_1(
    input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Compute the index of the current program.
    pid = tl.program_id(0)
    
    # Compute the start and end indices for this block.
    block_start = pid * BLOCK_SIZE
    block_end = tl.min(block_start + BLOCK_SIZE, n_elements)
    
    # Iterate over the elements in this block.
    for i in range(block_start, block_end):
        # Load the input element.
        input_val = tl.load(input_ptr + i)
        
        # Apply the isfinite function.
        result = tl.isfinite(input_val)
        
        # Store the result.
        tl.store(output_ptr + i, result)

def isfinite_func_wrapper_rank_1(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be CUDA tensors"
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    n_elements = input_tensor.numel()
    
    # Determine the block size. This can be tuned based on your hardware.
    BLOCK_SIZE = 1024
    
    # Calculate the number of blocks needed.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel.
    isfinite_func_kernel_rank_1[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
input_tensor = torch.tensor([1.0, float('inf'), 3.0, float('nan')], device='cuda')
output_tensor = torch.empty_like(input_tensor, dtype=torch.bool)

isfinite_func_wrapper_rank_1(input_tensor, output_tensor)

print(output_tensor)  # Output should be: tensor([True, False, True, False], device='cuda')
