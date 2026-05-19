import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale, Out,
    stride_qbs, stride_qh, stride_kbs, stride_kh,
    stride_vbs, stride_vh, stride_obs, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # Initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Compute pointer offsets
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh + offs_m[:, None]) * stride_qbs + offs_d[None, :]
    off_k = (cur_batch * stride_kbs + cur_head * stride_kh + offs_n[None, :]) * stride_kbs + offs_d[:, None]
    off_v = (cur_batch * stride_vbs + cur_head * stride_vh + offs_n[:, None]) * stride_vbs + offs_d[None, :]
    
    # Load Q, K, V
    q = tl.load(Q + off_q)
    k = tl.load(K + off_k)
    v = tl.load(V + off_v)

    # Load relative positional embeddings
    rel_pos_emb = tl.load(B0 + off_k)

    # Compute scaled dot-product attention
    qk = tl.dot(q, k) + rel_pos_emb
    qk *= sm_scale
    qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

    # Softmax
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    
    # Update output
    acc = tl.dot(p, v)
    
    # Store the result
    off_o = (cur_batch * stride_obs + cur_head * stride_oh + offs_m[:, None]) * stride_obs + offs_d[None, :]
    tl.store(Out + off_o, acc)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, sm_scale, batch, head, max_input_len):
    BLOCK = 128  # Define block size
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel_aligned[grid](
        q, k, v, b0, sm_scale, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1
    )
