import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from torch._inductor.triton_heuristics import grid
from torch._inductor import triton_helpers


@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_range = row_start + BLOCK_SIZE
    row_mask = row_start < n_elements

    cols = tl.arange(0, P2)
    col_mask = cols < BLOCK_SIZE

    x_ptr += row_start + cols
    output_ptr += row_start + cols
    output_maxs += row_start

    x_row = tl.load(x_ptr, mask=col_mask & row_mask, other=0.0, eviction_policy="evict_last")

    abs_x_row = tl.abs(x_row)
    max_val = tl.max(tl.where(col_mask, abs_x_row, 0), axis=0)
    tl.store(output_maxs, max_val, mask=row_mask)

    quantized_row = tl.libdevice.llrint(127.0 * (x_row / max_val))
    tl.store(output_ptr, quantized_row, mask=col_mask & row_mask)


def quantize_rowwise(x):
    assert x.is_cuda, "Input must be a CUDA tensor"

    n_elements = x.numel()
    output = x.new_empty(x.shape, dtype=torch.int8)
    output_maxs = x.new_empty((n_elements,), dtype=torch.float16)

    BLOCK_SIZE = triton_helpers.next_power_of_2(x.size(1))
    P2 = misc.find_power_of_2(BLOCK_SIZE)
    grid = lambda meta: (triton_helpers.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE, P2)
    return output, output_maxs
