import triton
import torch
import triton.language as tl

# Triton kernel for forward GEGLU operation
@triton.jit
def _geglu_tanh_forward_kernel(
    a,
    b,
    c,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_i in tl.range(row_start, n_rows, row_step):
        row_offset = row_i * n_cols
        col_offsets = row_i * BLOCK_SIZE
        a_ptrs = a + row_offset + col_offsets
        b_ptrs = b + row_offset + col_offsets
        c_ptrs = c + row_offset + col_offsets
        mask = col_offsets < n_cols

        a_val = tl.load(a_ptrs, mask=mask, other=0).to(tl.float32)
        b_val = tl.load(b_ptrs, mask=mask, other=0)

        s = 0.7978845608028654  # math.sqrt(2 / math.pi)
        u = a_val * s
        t = tl.math.tanh(u)
        f_row = 0.5 * a_val * (t + 1.0)
        h_row = b_val * f_row

        tl.store(c_ptrs, h_row, mask=mask)

# Function to perform forward computation with GEGLU
def geglu_forward(a, b):
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])

    assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]

    c = torch.empty_like(a)
    n_rows, n_cols = a.shape

    BLOCK_SIZE = 128
    num_warps = 4

    grid = lambda meta: (
        triton.cdiv(n_rows, meta["BLOCK_SIZE"]),
        triton.cdiv(meta["num_warps"], meta["BLOCK_SIZE"]),
    )
    _geglu_tanh_forward_kernel[grid](
        a, b, c, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return c

# Triton kernel for backward GEGLU operation
@triton.jit
def _geglu_tanh_backward_kernel(
    dc,
    a,
    b,
    da,
    db,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_i in tl.range(row_start, n_rows, row_step):
        row_offset = row_i * n_cols
        col_offsets = row_i * BLOCK_SIZE
        dc_ptrs = dc + row_offset + col_offsets
        a_ptrs = a + row_offset + col_offsets
        b_ptrs = b + row_offset + col_offsets
        da_ptrs = da + row_offset + col_offsets
        db_ptrs = db + row_offset + col_offsets
        mask = col_offsets < n_cols

        dc_val = tl.load(dc_ptrs, mask=mask, other=0)
        a_val = tl.load(a_ptrs, mask=mask, other=0).to(tl.float32)
        b_val = tl.load(b_ptrs, mask=mask, other=0)

        s = 0.7978845608028654  # math.sqrt(2 / math.pi)
        u = a_val * s
        t = tl.math.tanh(u)

        df_da = 0.5 * (1.0 + t)
        df_db = tl.load(b_ptrs, mask=mask, other=0)

        h_val = df_da * dc_val

        tl.store(da_ptrs, h_val, mask=mask)
        tl.store(db_ptrs, df_db * dc_val, mask=mask)

# Function to perform backward computation with GEGLU
def geglu_backward(dc, a, b):
    dc = dc.detach()
    a = a.detach()

    da = torch.empty_like(a)
    db = torch.empty_like(b)

    assert dc.shape[0] == a.shape[0] and dc.shape[0] == b.shape[0]
    assert dc.shape[1] == a.shape[1] and dc.shape[1] == b.shape[1]

    n_rows, n_cols = dc.shape

    BLOCK_SIZE = 128
    num_warps = 4

    grid = lambda meta: (
        triton.cdiv(n_rows, meta["BLOCK_SIZE"]),
        triton.cdiv(meta["num_warps"], meta["BLOCK_SIZE"]),
    )
    _geglu_tanh_backward_kernel[grid](
        dc,
        a,
        b,
        da,
        db,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return da, db
