import triton
import triton.language as tl

@triton.jit
def index_fill_kernel(
    input_ptr,       # Pointer to the input tensor
    index_ptr,       # Pointer to the index tensor
    output_ptr,      # Pointer to the output tensor
    input_shape,     # Shape of the input tensor
    index_shape,     # Shape of the index tensor
    dim,             # Dimension along which to index
    value,           # Value to fill with
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    # Get program id and indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    indices = index_ptr[block_start:block_start + BLOCK_SIZE]
    
    # Iterate over the elements in the block
    for i in range(BLOCK_SIZE):
        idx = indices[i]
        if idx < input_shape[dim]:
            # Calculate the linear offset for the output tensor
            linear_idx = 0
            for j in range(dim):
                linear_idx += input_ptr[idx][j] * input_shape[j+1:]
            output_ptr[linear_idx] = value
