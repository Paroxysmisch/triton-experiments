@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Get program ids for x and y axes
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    # Calculate row and column offsets
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_mask = cols_offsets < N

    # Compute masks for blocks and output
    block_mask = rows_mask & cols_mask
    out_mask = rows_mask & (cols_offsets < index_len)

    # Load indices and compute offsets
    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    # Load selected input and store in output
    selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)
