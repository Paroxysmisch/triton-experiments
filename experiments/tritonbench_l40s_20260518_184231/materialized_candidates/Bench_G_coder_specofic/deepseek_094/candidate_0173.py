import triton
import numpy as np

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr,
    index_ptr,
    output_ptr,
    num_indices,
    num_cols,
    BLOCK_SIZE_INDEX=1024,
    BLOCK_SIZE_COL=1024,
):
    pid0 = triton.program_id(axis=0)
    pid1 = triton.program_id(axis=1)

    # compute offsets for source and output tensors
    source_offset = pid0 * BLOCK_SIZE_INDEX * num_cols + pid1 * BLOCK_SIZE_COL
    output_offset = pid0 * BLOCK_SIZE_INDEX + pid1 * BLOCK_SIZE_COL

    # apply masks to ensure bounds
    source_mask = triton.mask(pid0 < num_indices and pid1 < num_cols)
    output_mask = triton.mask(pid0 < num_indices)

    # fetch data from source and write into output tensor
    for i in range(BLOCK_SIZE_INDEX):
        for j in range(BLOCK_SIZE_COL):
            source_val = triton.load(source_ptr + source_offset + i * num_cols + j, mask=source_mask)
            triton.store(output_ptr + output_offset + i + j, source_val, mask=output_mask)

def index_select_cat_fwd(source, index):
    source_ptr = triton.ptr(source)
    index_ptr = triton.ptr(index)
    output_ptr = triton.ptr(np.empty_like(source))

    num_indices = index.shape[0]
    num_cols = source.shape[1]

    # compute dimensions of the launch grid
    grid = lambda meta: (triton.cdiv(num_indices, meta['BLOCK_SIZE_INDEX']), triton.cdiv(num_cols, meta['BLOCK_SIZE_COL']))

    # launch the kernel
    index_select_cat_fwd_kernel[grid](source_ptr, index_ptr, output_ptr, num_indices, num_cols)

    return output_ptr
