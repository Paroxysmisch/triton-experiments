import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source,
    index,
    out,
    M,
    N,
    index_len,
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    index_offsets = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    index_mask = index_offsets < index_len
    source_offsets = tl.load(index + index_offsets, mask=index_mask, other=0)
    source_mask = (index_mask[:, None] & (tl.arange(0, BLOCK_SIZE_COL)[None, :] < N))

    block = tl.load(
        source + source_offsets[:, None] * N + tl.arange(0, BLOCK_SIZE_COL)[None, :],
        mask=source_mask,
        other=0.0,
    )
    out_offsets = (
        pid0 * BLOCK_SIZE_INDEX * N
        + index_offsets[:, None] * N
        + tl.arange(0, BLOCK_SIZE_COL)[None, :]
    )
    out_mask = index_mask[:, None] & (
        (tl.arange(0, BLOCK_SIZE_COL)[None, :] + pid0 * BLOCK_SIZE_INDEX < M)
    )
    tl.store(out + out_offsets, block, mask=out_mask)


def index_select_cat_fwd(source, index):
    assert index.ndim == 1, "Index should have dimension 1"
    assert source.ndim == 2, "Source should have dimension 2"
    M, N = source.shape
    index_len = index.numel()
    if index_len > M:
        print(
            f"Warning: Index length ({index_len}) is greater than source length ({M}). Truncating index."
        )
        index = index[:M]
        index_len = M
    out = torch.empty((M, index_len), dtype=source.dtype, device=source.device)
    source_stride_0 = source.stride(0)
    source_stride_1 = source.stride(1)
    index_stride_0 = index.stride(0)
    BLOCK_SIZE_INDEX = 32
    BLOCK_SIZE_COL = 32

    grid = lambda meta: (
        triton.cdiv(index_len, meta["BLOCK_SIZE_INDEX"]),
        meta["BLOCK_SIZE_COL"],
    )

    index_select_cat_fwd_kernel[grid](
        source,
        index,
        out,
        M,
        N,
        index_len,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    out_stride_0 = out.stride(0)
    out_stride_1 = out.stride(1)
    return out
