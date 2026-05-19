import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_fwd_kernel(
    source, output, index, M, N, stride_idx, BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    src_start = pid0 * BLOCK_SIZE_INDEX
    index_start = pid1 * BLOCK_SIZE_COL
    src_end = tl.minimum((pid0 + 1) * BLOCK_SIZE_INDEX, M)
    index_end = tl.minimum((pid1 + 1) * BLOCK_SIZE_COL, N)

    src_off = src_start * N + index_start
    index_off = index_start
    output_off = (src_start * BLOCK_SIZE_COL + index_start)
    
    mask = (src_off < M * N) & (index_off < N)
    
    src_index = (index + index_off).to(tl.int32)

    for _ in range(src_start, src_end):
        src_pointer = source + (src_index * stride_idx)
        selected = tl.load(src_pointer, mask=mask, other=0.0)
        tl.store(output + output_off, selected, mask=mask)
        src_off += N
        src_index += 1
        output_off += BLOCK_SIZE_COL

def index_select_cat_fwd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    assert source.is_cuda and index.is_cuda
    assert source.ndim == 2
    assert index.ndim == 1
    M, N = source.shape
    stride_idx = source.stride(0)
    index_len = index.size(0)
    
    output = torch.empty((M, index_len), device="cuda", dtype=source.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_INDEX"]), triton.cdiv(N, meta["BLOCK_SIZE_COL"]), )
    
    index_select_cat_fwd_kernel[grid](
        source,
        output,
        index,
        M,
        N,
        stride_idx
    )
    
    if index_len > M:
        warnings.warn(f"The number of indices ({index_len}) exceeds the number of rows ({M}) in the source. The indices are truncated.")
    
    return output
