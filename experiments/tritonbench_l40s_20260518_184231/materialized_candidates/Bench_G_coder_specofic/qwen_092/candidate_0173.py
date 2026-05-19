import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source: tl.tensor,  # 2D tensor (source)
    index: tl.tensor,   # 1D tensor (indices)
    output: tl.tensor,  # 2D tensor (output)
    source_rows: tl.int32,
    source_cols: tl.int32,
    index_len: tl.int32,
    BLOCK_SIZE_INDEX: tl.constexpr(128),
    BLOCK_SIZE_COL: tl.constexpr(32),
):
    # Get the program ids
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Compute the starting index and column for the block
    start_index = pid0 * BLOCK_SIZE_INDEX
    start_col = pid1 * BLOCK_SIZE_COL

    # Compute the number of indices and columns this block will handle
    indices_per_block = min(BLOCK_SIZE_INDEX, index_len - start_index)
    cols_per_block = min(BLOCK_SIZE_COL, source_cols - start_col)

    # Iterate over the indices and columns in this block
    for i in range(indices_per_block):
        idx = start_index + i
        if idx >= index_len:
            break
        row_idx = index[idx]
        if row_idx >= source_rows:
            continue

        for j in range(cols_per_block):
            col_idx = start_col + j
            if col_idx >= source_cols:
                break

            # Compute the source and output offsets
            source_offset = row_idx * source_cols + col_idx
            output_offset = idx * source_cols + col_idx

            # Load data from source and store it in output
            value = tl.load(source[source_offset])
            tl.store(output[output_offset], value)
