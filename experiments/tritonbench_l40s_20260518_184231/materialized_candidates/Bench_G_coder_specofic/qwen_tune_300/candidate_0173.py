import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source,
    index,
    out,
    s_src_0: int, s_src_1: int,
    s_out_0: int, s_out_1: int,
    n_src_rows: int, n_src_cols: int,
    n_index: int,
    n_out_rows: int,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    # compute offset
    src_offs = tl.arange(0, BLOCK_SIZE_COL) + pid1 * BLOCK_SIZE_COL
    out_offs = tl.arange(0, BLOCK_SIZE_COL) + pid1 * BLOCK_SIZE_COL
    src_mask = src_offs < n_src_cols
    out_mask = out_offs < n_src_cols

    cols = src_offs + (pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)) * n_src_cols

    ind_mask = (pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)) < n_index

    index_offs = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    index_mask = index_offs < n_index

    index_vals = tl.load(index + index_offs, mask=ind_mask, other=0)

    src_offs = index_vals[:, None] * s_src_0 + cols[None, :] * s_src_1
    out_offs = cols[None, :] * s_out_1

    src_ptrs = source + src_offs
    out_ptrs = out + out_offs

    src_vals = tl.load(src_ptrs, mask=ind_mask[:, None] & src_mask[None, :], other=0.0)
    tl.store(out_ptrs, src_vals, mask=out_mask[None, :])


def index_select_cat_fwd(source, index):
    if source.is_cuda and index.is_cuda:
        assert source.dim() == 2 and index.dim() == 1
        if index.numel() > source.numel():
            print("Warning: index numel > source numel, truncating index to source numel")
            index = index[: source.numel()]
        out = torch.empty((index.numel(), source.size(1)), dtype=source.dtype, device=source.device)
        s_src_0, s_src_1 = source.stride()
        s_out_0, s_out_1 = out.stride()
        n_src_rows, n_src_cols = source.size()
        n_index = index.size(0)
        n_out_rows = out.size(0)
        BLOCK_SIZE_INDEX = 16
        BLOCK_SIZE_COL = 32
        grid = (
            triton.cdiv(n_index, BLOCK_SIZE_INDEX),
            triton.cdiv(n_src_cols, BLOCK_SIZE_COL),
        )
        index_select_cat_fwd_kernel[grid](
            source,
            index,
            out,
            s_src_0,
            s_src_1,
            s_out_0,
            s_out_1,
            n_src_rows,
            n_src_cols,
            n_index,
            n_out_rows,
            BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
            BLOCK_SIZE_COL=BLOCK_SIZE_COL,
        )
        return out
    else:
        raise NotImplementedError
