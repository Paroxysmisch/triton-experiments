['BLOCK_SIZE']),)
    geglu_exact_backward_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE=1024)
    return DW, e, g

# Triton kernel for approximate forward GEGLU operation
@triton.jit
def _approx_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)

# Python function that wraps the approximate forward kernel
def geglu_approx_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device="cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _approx_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=1024)
    return out

# Triton kernel for approximate backward GEGLU operation
@triton.jit
def _approx_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_partial_row = e_row * tl.sigmoid(e_row)
    df_de = (1.0 + f_partial_row) * tl.sigmoid(e_row)
    df_row = DW_row * f_partial_row
    dg_row = DW_row * g_row

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    tl.store(DW + offsets, df_row, mask=mask)
    tl.store(e + offsets, dg_row, mask=mask)
    tl.store(g + offsets, de_row, mask=mask)

# Python function that wraps the approximate backward kernel
def geglu_approx_backward_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    geglu_approx_backward_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE=1024)
    return DW, e, g
