import torch
import triton
import triton.language as tl
from triton.runtime.jit import get_cuda_stream

@triton.jit
def _normalize_kernel(
    x_ptr, out_ptr,
    row_stride, n_cols,
    p_norm, eps_norm,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols

    # Load row
    elements = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute sum(|elements|^
