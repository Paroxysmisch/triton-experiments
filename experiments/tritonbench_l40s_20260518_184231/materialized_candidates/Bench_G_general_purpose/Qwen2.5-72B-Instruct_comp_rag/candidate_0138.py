import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_d_blocks: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    block_m = tl.program_id(2)

    kv_head = head // num_kv_heads

    # Calculate the start and end indices for the current block
    block_start = layout_csr_row_indices[batch * num_heads + head] + block_m * BLOCK_M
    block_end = layout_csr_row_indices[batch * num_heads + head + 1]

    # Initialize offsets
    offs_m = block_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Load the query block
    off_q = (batch * stride_qbs + head * stride_qh + offs_m[:, None] * BLOCK_D + offs_d[None, :])
    q = tl.load(Q + off_q, mask=offs_m[:, None] < block_end, other=0.0)

    # Initialize the output block
    out_block = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over the non-zero blocks in the CSR format
    for block_n in range(layout_csr_row_indices[batch * num_heads + head], layout_csr_row_indices[batch * num_heads + head + 1]):
        col_start = layout_csr_col_indices[block_n] * BLOCK_N
        col_end = col_start + BLOCK_N

        # Load the key and value blocks
        off_k = (batch * stride_kbs + kv_head * stride_kh + col_start + offs_n[None, :] * BLOCK_D + offs_d[:, None])
        k = tl.load(K + off_k, mask=col_start + offs_n[None, :] < block_end, other=0.0)

        off_v = (batch * stride_vbs + kv_head * stride_vh + col_start + offs_n[:, None] * BLOCK_D + offs_d[None, :])
        v = tl.load(V + off_v, mask=col_start + offs_n[:, None] < block_end, other=0.0)

        # Compute the dot product and scale
        qk = tl.dot(q, k)
        qk *= softmax_scale

        # Compute the softmax
        m_i = tl.max(qk, 1)
        p = tl.exp(qk - m_i[:, None])
        l_i = tl.sum(p, 1)
        p /= l_i[:, None]

        # Accumulate the output
        out_block += tl.dot(p, v)

    # Store the output block
    off_o = (batch * stride_obs + head * stride_oh + offs_m[:, None] * BLOCK_D + offs_d[None, :])
    tl.store(out + off_o, out_block, mask=offs_m[:, None] < block_end)

def block_sparse_attention(q, k, v, layout_csr_row_indices, layout_csr_col_indices, out, num_heads, num_kv_heads, num_d_blocks, max_input_len, softmax_scale):
    if CUDA_CAPABILITY[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    batch, head, seq_len = q.shape[0], q.shape[1], q.shape[2]
    grid = (batch, head, triton.cdiv(seq_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    block_sparse_attention_kernel[grid](
        q,
        k,
        v,
        layout_csr_row_indices,
        layout_csr_col_indices,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_d_blocks=num_d_blocks,
        softmax_scale=softmax_scale,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_D=Lk,
        num_warps=num_warps,
        num_stages=1,
    )
