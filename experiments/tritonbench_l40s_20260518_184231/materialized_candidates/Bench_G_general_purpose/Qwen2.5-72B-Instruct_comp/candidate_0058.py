import triton
import triton.language as tl

# Heuristic functions to determine optimal tile sizes and number of warps
def heuristics_for_tile_size(input_size):
    if input_size < 1024:
        return 128
    elif input_size < 4096:
        return 256
    else:
        return 512

def heuristics_for_num_warps(input_size):
    if input_size < 1024:
        return 1
    elif input_size < 4096:
        return 2
    else:
        return 4

# Triton kernel for rank-1 tensors
@triton.jit
def isfinite_func_kernel_rank_1(in_ptr, out_ptr, n_elements, one_tile_per_cta, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in0 = tl.load(in_ptr + offsets, mask=mask)
    out0 = tl.where(tl.math.isfinite(in0), 1, 0)

    tl.store(out_ptr + offsets, out0, mask=mask)

# Wrapper function for the Triton kernel
def isfinite_func_wrapper_rank_1(input_tensor, output_tensor):
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"
    
    input_size = input_tensor.shape[0]
    tile_size = heuristics_for_tile_size(input_size)
    num_warps = heuristics_for_num_warps(input_size)
    
    num_ctas = (input_size + tile_size - 1) // tile_size
    grid = (num_ctas, 1, 1)
    
    one_tile_per_cta = tile_size == input_size
    
    isfinite_func_kernel_rank_1[grid](input_tensor, output_tensor, input_size, one_tile_per_cta, BLOCK_SIZE=tile_size, num_warps=num_warps)

# Example usage
import torch

# Create input and output tensors
input_tensor = torch.tensor([1.0, 2.0, float('inf'), float('-inf'), float('nan')], device='cuda')
output_tensor = torch.zeros_like(input_tensor, dtype=torch.int32, device='cuda')

# Call the wrapper function
isfinite_func_wrapper_rank_1(input_tensor, output_tensor)

# Print the result
print(output_tensor)
