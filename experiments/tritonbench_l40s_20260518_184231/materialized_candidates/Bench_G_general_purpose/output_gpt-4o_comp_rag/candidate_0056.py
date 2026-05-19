import triton
import triton.language as tl
import torch

# Define the finiteness check function for different data types
@triton.jit
def _isfinite(x):
    return tl.math.isfinite(x)

# Kernel function to check finiteness of elements in a rank-1 tensor
@triton.jit
def isfinite_func_kernel_rank_1(
    in_ptr, out_ptr, N, 
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block index and offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from input tensor
    mask = offsets < N
    in0 = tl.load(in_ptr + offsets, mask=mask)
    
    # Apply the finiteness check
    is_finite = _isfinite(in0)
    
    # Store the results in the output tensor
    tl.store(out_ptr + offsets, is_finite, mask=mask)

# Heuristic functions for determining tile size and number of warps
def heuristics_for_tile_size(N):
    # Heuristic to determine the tile size
    return 128  # You can adjust this based on your hardware

def heuristics_for_num_warps(N):
    # Heuristic to determine the number of warps
    return 4  # You can adjust this based on your hardware

# Wrapper function to process input and output tensors
def isfinite_func_wrapper_rank_1(input_tensor):
    # Ensure the input is a 1D tensor
    assert input_tensor.ndim == 1, "Input tensor must be rank-1"
    
    # Get the size of the input tensor
    N = input_tensor.shape[0]
    
    # Allocate output tensor
    output_tensor = torch.empty_like(input_tensor, dtype=torch.bool)
    
    # Determine tile size and number of warps
    BLOCK_SIZE = heuristics_for_tile_size(N)
    num_warps = heuristics_for_num_warps(N)
    
    # Calculate grid size
    num_ctas = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    grid = (num_ctas,)
    isfinite_func_kernel_rank_1[grid](input_tensor, output_tensor, N, BLOCK_SIZE)
    
    return output_tensor

# Example usage
input_tensor = torch.tensor([1.0, float('inf'), 3.0, float('nan')], device='cuda')
output_tensor = isfinite_func_wrapper_rank_1(input_tensor)
print(output_tensor)  # Should print tensor([ True, False,  True, False], device='cuda:0')
