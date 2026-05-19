import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptr, grad_output_ptr, index_ptr,
    grad_source_stride0, grad_source_stride1,
    grad_output_stride0, grad_output_stride1,
    index_stride0, index_stride1,
    num_indices, BLOCK_SIZE: tl.constexpr
):
    # Define the block indices
    block_idx = tl.program_id(0)
    
    # Compute the row start for this block
    row_start = block_idx * BLOCK_SIZE
    
    # Iterate over the block size
    for i in range(BLOCK_SIZE):
        # Calculate the current row
        row = row_start + i
        
        # Check if within bounds
        if row < num_indices:
            # Load the index
            index = tl.load(index_ptr + row * index_stride0)
            
            # Load the grad_output value
            grad_output_val = tl.load(grad_output_ptr + row * grad_output_stride0)
            
            # Compute the position in grad_source to update
            grad_source_pos = index * grad_source_stride0
            
            # Load the current value in grad_source
            grad_source_val = tl.load(grad_source_ptr + grad_source_pos)
            
            # Update grad_source with the value from grad_output
            tl.store(grad_source_ptr + grad_source_pos, grad_source_val + grad_output_val)

def index_select_cat_bwd(grad_source, index, grad_output):
    # Ensure inputs are on CUDA
    assert grad_source.is_cuda and index.is_cuda and grad_output.is_cuda, "All inputs must be CUDA tensors"
    
    # Check dimensions and strides
    assert grad_source.dim() == 2 and index.dim() == 2 and grad_output.dim() == 2, "All inputs must be 2D tensors"
    assert grad_source.stride(0) == grad_output.stride(0), "grad_source and grad_output must have the same stride for dim 0"
    
    # Get tensor strides
    grad_source_stride0, grad_source_stride1 = grad_source.stride()
    grad_output_stride0, grad_output_stride1 = grad_output.stride()
    index_stride0, index_stride1 = index.stride()
    
    # Number of indices
    num_indices = index.size(0)
    
    # Define block size
    BLOCK_SIZE = 128  # Can be tuned based on the GPU architecture
    
    # Launch the Triton kernel
    grid = (triton.cdiv(num_indices, BLOCK_SIZE),)
    index_select_cat_bwd_kernel[grid](
        grad_source, grad_output, index,
        grad_source_stride0, grad_source_stride1,
        grad_output_stride0, grad_output_stride1,
        index_stride0, index_stride1,
        num_indices, BLOCK_SIZE
    )

# Example usage:
grad_source = torch.zeros((10, 10), device='cuda')
index = torch.tensor([[0], [1], [2]], device='cuda', dtype=torch.int32)
grad_output = torch.ones((3, 10), device='cuda')

index_select_cat_bwd(grad_source, index, grad_output)
print(grad_source)
