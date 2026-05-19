import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr,  # Pointer to the gradient of the output tensor
    grad_source_ptr,  # Pointer to the gradient of the source tensor
    index_ptr,        # Pointer to the index tensor
    grad_output_stride0,  # Stride of the output gradient tensor along the first dimension
    grad_output_stride1,  # Stride of the output gradient tensor along the second dimension
    grad_source_stride0,  # Stride of the source gradient tensor along the first dimension
    grad_source_stride1,  # Stride of the source gradient tensor along the second dimension
    index_stride0,        # Stride of the index tensor along the first dimension
    num_indices,          # Number of indices
    num_cols,             # Number of columns in the tensors
    BLOCK_SIZE_INDEX: tl.constexpr,  # Block size for the index dimension
    BLOCK_SIZE_COL: tl.constexpr     # Block size for the column dimension
):
    pid = tl.program_id(axis=0)
    num_blocks_index = (num_indices + BLOCK_SIZE_INDEX - 1) // BLOCK_SIZE_INDEX
    num_blocks_col = (num_cols + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL
    block_id_index = pid // num_blocks_col
    block_id_col = pid % num_blocks_col

    start_index = block_id_index * BLOCK_SIZE_INDEX
    start_col = block_id_col * BLOCK_SIZE_COL

    offsets_index = start_index + tl.arange(0, BLOCK_SIZE_INDEX)
    offsets_col = start_col + tl.arange(0, BLOCK_SIZE_COL)

    mask_index = offsets_index < num_indices
    mask_col = offsets_col < num_cols

    for i in range(BLOCK_SIZE_INDEX):
        if mask_index[i]:
            index = tl.load(index_ptr + offsets_index[i] * index_stride0)
            for j in range(BLOCK_SIZE_COL):
                if mask_col[j]:
                    grad_output_offset = index * grad_output_stride0 + offsets_col[j] * grad_output_stride1
                    grad_source_offset = offsets_index[i] * grad_source_stride0 + offsets_col[j] * grad_source_stride1
                    grad_output_value = tl.load(grad_output_ptr + grad_output_offset)
                    tl.atomic_add(grad_source_ptr + grad_source_offset, grad_output_value)

import torch
import triton
import triton.language as tl

def index_select_cat_bwd(grad_output, index, grad_source):
    # Check tensor properties
    assert grad_output.is_cuda, "grad_output must be a CUDA tensor"
    assert grad_source.is_cuda, "grad_source must be a CUDA tensor"
    assert index.is_cuda, "index must be a CUDA tensor"
    assert grad_output.dim() == 2, "grad_output must be a 2D tensor"
    assert grad_source.dim() == 2, "grad_source must be a 2D tensor"
    assert index.dim() == 1, "index must be a 1D tensor"
    assert grad_output.size(1) == grad_source.size(1), "grad_output and grad_source must have the same number of columns"
    assert grad_output.size(0) == index.size(0), "grad_output and index must have the same number of rows"

    # Get tensor dimensions
    num_indices = index.size(0)
    num_cols = grad_output.size(1)

    # Define block sizes
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 32

    # Define grid size
    grid = (triton.cdiv(num_indices, BLOCK_SIZE_INDEX) * triton.cdiv(num_cols, BLOCK_SIZE_COL),)

    # Zero out the gradient of the source tensor
    grad_source.zero_()

    # Launch the kernel
    index_select_cat_bwd_kernel[grid](
        grad_output.data_ptr(),
        grad_source.data_ptr(),
        index.data_ptr(),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_source.stride(0),
        grad_source.stride(1),
        index.stride(0),
        num_indices,
        num_cols,
        BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL
    )

    return grad_source
