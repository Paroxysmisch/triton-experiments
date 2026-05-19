import torch
import triton
import triton.language as tl

@triton.jit
def index_select_kernel(
    # Pointers to tensors
    input_ptr,    # pointer to input tensor
    output_ptr,   # pointer to output tensor
    index_ptr,    # pointer to index tensor
    # Shapes
    n_rows,       # number of rows in input
    n_cols,       # number of columns in input
    index_size,   # size of index tensor
    # Block sizes (compile-time constants)
    BLOCK_SIZE_INDEX: tl.constexpr,  
    BLOCK_SIZE_COL: tl.constexpr,
):
    # Compute program ID
    pid_i = tl.program_id(0)  # index dimension
    pid_c = tl.program_id(1)  # column dimension
    
    # Calculate offsets
    index_offset = pid_i * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    col_offset = pid_c * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    
    # Create masks
    index_mask = index_offset < index_size
    col_mask = col_offset < n_cols
    
    # Load indices
    indices = tl.load(index_ptr + index_offset, mask=index_mask, other=0)
    
    # For each valid index, load corresponding row from input
    for idx in range(0, BLOCK_SIZE_INDEX):
        if idx < index_size:
            # Calculate input offset
            row_idx = indices[idx]
            in_offset = row_idx * n_cols + col_offset
            
            # Load input data
            x = tl.load(input_ptr + in_offset, mask=col_mask, other=0.0)
            
            # Calculate output offset
            out_offset = idx * n_cols + col_offset
            
            # Store to output
            tl.store(output_ptr + out_offset, x, mask=col_mask)

def index_select(input_tensor, index):
    # Get tensor dimensions
    n_rows = input_tensor.shape[0]
    n_cols = input_tensor.shape[1]
    index_size = index.shape[0]
    
    # Create output tensor
    output = torch.empty((index_size, n_cols), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Define block sizes
    BLOCK_SIZE_INDEX = 32
    BLOCK_SIZE_COL = 32
    
    # Calculate grid
    grid = (
        triton.cdiv(index_size, BLOCK_SIZE_INDEX),
        triton.cdiv(n_cols, BLOCK_SIZE_COL),
    )
    
    # Launch kernel
    index_select_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        index_ptr=index,
        n_rows=n_rows,
        n_cols=n_cols,
        index_size=index_size,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    
    return output
