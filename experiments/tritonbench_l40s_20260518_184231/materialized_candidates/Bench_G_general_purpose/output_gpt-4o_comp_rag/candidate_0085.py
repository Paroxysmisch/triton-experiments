import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr, grad_source_ptr, index_ptr, num_indices, num_columns,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    # Compute program ID for indices and columns
    pid_index = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Compute the offset for the current block
    index_offsets = pid_index * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    col_offsets = pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    # Mask to prevent out-of-bounds memory accesses
    index_mask = index_offsets < num_indices
    col_mask = col_offsets < num_columns

    # Load indices
    indices = tl.load(index_ptr + index_offsets, mask=index_mask, other=0)

    # Compute the offsets for grad_output and grad_source
    grad_output_offsets = indices[:, None] * num_columns + col_offsets[None, :]
    grad_source_offsets = index_offsets[:, None] * num_columns + col_offsets[None, :]

    # Load gradients from grad_output
    grad_output = tl.load(grad_output_ptr + grad_output_offsets, mask=index_mask[:, None] & col_mask[None, :], other=0)

    # Store gradients into grad_source
    tl.store(grad_source_ptr + grad_source_offsets, grad_output, mask=index_mask[:, None] & col_mask[None, :])

def index_select_cat_bwd(grad_output, index, num_indices, num_columns):
    # Ensure the inputs are CUDA tensors
    assert grad_output.is_cuda and index.is_cuda, "Inputs must be CUDA tensors"
    assert grad_output.dim() == 2 and index.dim() == 1, "grad_output must be 2D and index must be 1D"
    
    # Create an empty tensor for the gradients of the source
    grad_source = torch.zeros((num_indices, num_columns), dtype=grad_output.dtype, device=grad_output.device)

    # Define block sizes
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 128

    # Define grid size
    grid = lambda meta: (
        triton.cdiv(num_indices, meta['BLOCK_SIZE_INDEX']),
        triton.cdiv(num_columns, meta['BLOCK_SIZE_COL'])
    )

    # Launch the kernel
    index_select_cat_bwd_kernel[grid](
        grad_output, grad_source, index, num_indices, num_columns,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX, BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

    return grad_source
