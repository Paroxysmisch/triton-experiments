import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr, index_ptr, output_ptr, M, N, BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE_COL)
    index_base = pid0 * BLOCK_SIZE_INDEX
    index_offsets = index_base + tl.arange(0, BLOCK_SIZE_INDEX)
    output_row_base = pid1 * BLOCK_SIZE_INDEX
    output_row_offsets = output_row_base + tl.arange(0, BLOCK_SIZE_INDEX)
    output_row_mask = output_row_offsets < M
    index_mask = index_offsets < M
    source_ptr += index_offsets[:, None] * source_ptr.strides[0] + col_offsets[None, :] * source_ptr.strides[1]
    output_ptr += output_row_offsets[:, None] * output_ptr.strides[0] + col_offsets[None, :] * output_ptr.strides[1]
    indices = tl.load(index_ptr + index_offsets, mask=index_mask)
    if not (BLOCK_SIZE_INDEX == 1 and BLOCK_SIZE_COL == 1):
        tl.debug_barrier()
    for _ in range(0, 1 if BLOCK_SIZE_COL == 1 else indices.size(1)):
        data = tl.load(source_ptr, mask=output_row_mask, other=0.0)
        tl.store(output_ptr, data, mask=output_row_mask)

def index_select_cat_fwd(source: torch.Tensor, index: torch.Tensor, dim=0):
    M = index.size(0)
    N = source.size(1) if source.dim() == 2 else 1
    if M > source.size(0):
        warnings.warn("number of indices exceeds the number of rows in the source")
        M = source.size(0)
    index = index.contiguous()
    index_dtype = index.dtype
    output = torch.empty((M, N), device=source.device, dtype=source.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_INDEX"]),
        triton.cdiv(N, meta["BLOCK_SIZE_COL"]),
    )
    index_select_cat_fwd_kernel[grid](source, index, output, M, N)
    return output.to(index_dtype)
