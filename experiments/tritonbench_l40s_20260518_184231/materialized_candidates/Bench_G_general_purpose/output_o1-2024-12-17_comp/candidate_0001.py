import triton
import triton.language as tl
import torch

@triton.jit
def _dequantize_rowwise(
    x_ptr,         # *int8
    state_x_ptr,   # *fp32
    output_ptr,    # *fp32
    inv_127,       # fp32
    n_elements,    # int32
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
    **meta
):
    pid = tl.program_id(0)
    row_offset = pid * P2
    cols = meta["n_cols"]
    col_ids = tl.arange(0, P2)
    offsets = row_offset + col_ids

    mask = offsets < (pid * cols + cols)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    max_val = tl.load(state_x_ptr + pid)
    deq_vals = (x_vals * max_val) * inv_127

    tl.store(output_ptr + offsets, deq_vals, mask=mask)

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    n_rows, n_cols = x.shape
    # Ensure CUDA
    assert x.is_cuda and state_x.is_cuda, "Input tensors must be on CUDA."

    # Prepare output
    output = torch.empty_like(x, dtype=torch.float32)

    # Next power of two >= n_cols
    P2 = 1
    while P2 < n_cols:
        P2 <<= 1

    grid = (n_rows,)
    block_size = 1024  # example block size
    inv_127 = 1.0 / 127.0

    _dequantize_rowwise[grid](
        x, 
        state_x, 
        output, 
        inv_127, 
        x.numel(),
        BLOCK_SIZE=block_size,
        P2=P2,
        n_cols=n_cols
    )
    return output
