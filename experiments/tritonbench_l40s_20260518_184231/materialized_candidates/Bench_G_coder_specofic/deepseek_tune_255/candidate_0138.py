import triton
import triton.language as tl
import torch

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
    out, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
    num_queries: tl.constexpr, num_kv_pairs: tl.constexpr,
    num_heads: tl.constexpr,
    head_group_size: tl.constexpr,
    CACHE_KEY_SELF_ATTN: tl.constexpr,
    kv_group_size: tl.constexpr = 1,
    stride_qm: tl.constexpr = 0, stride_qn: tl.constexpr = 0, stride_qd: tl.constexpr = 0,
    stride_qh: tl.constexpr = 0,
    stride_km: tl.constexpr = 0, stride_kn: tl.constexpr = 0, stride_kd: tl.constexpr = 0,
    stride_kh: tl.constexpr = 0,
    stride_vm: tl.constexpr = 0, stride_vn: tl.constexpr = 0, stride_vd: tl.constexpr = 0,
    stride_vh: tl.constexpr = 0,
    stride_layout_csr_row: tl.constexpr = 0, stride_layout_csr_col: tl.constexpr = 0,
    stride_om: tl.constexpr = 0, stride_on: tl.constexpr = 0, stride_od: tl.constexpr = 0,
    stride_oh: tl.constexpr = 0,
    PAST_SEQ_LEN: tl.constexpr = 0,
    RIGHT_PAD: tl.constexpr = True,
    D_HEAD: tl.constexpr = 0,
    NO_GROUP: tl.constexpr = False,
    NUM_KV_HEADS: tl.constexpr = 0,
    BLOCK_M_TILED: tl.constexpr = False,
    BLOCK_N_TILED: tl.constexpr = False,
    SPLIT_K: tl.constexpr = False,
    SPLIT_V: tl.constexpr = False,
    BIAS_SHIFT: tl.constexpr = 0,
    SPLIT_KV: tl.constexpr = False,
    BLOCK_M_TILED_2: tl.constexpr = False,
    BLOCK_N_TILED_2: tl.constexpr = False,
    HAS_BIAS: tl.constexpr = False,
    BIAS_ROW_MAJOR: tl.constexpr = False,
    IS_CAUSAL: tl.constexpr = False,
    MASK_STEPS: tl.constexpr = 0,
    D_HEAD_LT_32: tl.constexpr = False,
    ALLOW_PARALLEL_REDUCE: tl.constexpr = False,
    BLOCK_M_TILED_4: tl.constexpr = False,
    BLOCK_N_TILED_4: tl.constexpr = False,
):
    start_m = tl.program_id(0)
    if BLOCK_M_TILED:
        start_m *= BLOCK_M
    start_n = tl.program_id(1)
    if BLOCK_N_TILED:
        start_n *= BLOCK_N
    cur_device = tl.program_id(2)
    cur_batch = tl.program_id(3)
    cur_head = tl.program_id(4)

    if not NO_GROUP:
        cur_head %= head_group_size
        num_heads = head_group_size

    if NUM_KV_HEADS > 0:
        cur_head %= NUM_KV_HEADS
        num_heads = NUM_KV_HEADS

    if not D_HEAD_LT_32:
        d_head = D_HEAD
    else:
        d_head = tl.load(layout_csr_col_indices + cur_head * stride_layout_csr_col)

    if SPLIT_KV:
        kv_group_size = kv_group_size or head_group_size

    q_offset = cur_batch * stride_qm * num_queries + start_m * stride_qn + cur_head * stride_qh
    if not NO_GROUP:
        q_offset += cur_head * d_head * stride_qd

    kv_offset = cur_head * stride_kn
    if not NO_GROUP:
        kv_offset += cur_head * d_head * stride_kd

    bias_offset = cur_head * stride_kn
    if not NO_GROUP:
        bias_offset += cur_head * d_head * stride_kd

    if SPLIT_KV:
        k_offset = kv_offset
        v_offset = kv_offset + num_kv_pairs * stride_kn
        if not NO_GROUP:
            k_offset += cur_head * d_head * stride_kd
            v_offset += cur_head * d_head * stride_vd
    else:
        k_offset = kv_offset
        v_offset = kv_offset
        if not NO_GROUP:
            k_offset += cur_head * d_head * stride_kd
            v_offset += cur_head * d_head * stride_vd

    q = tl.load(Q + q_offset)

    if HAS_BIAS:
        if BIAS_ROW_MAJOR:
            bias = tl.load(b + bias_offset)
        else:
            bias = tl.load(b + bias_offset // 8 + (bias_offset % 8) * stride_b)
        q = (q >> BIAS_SHIFT) + bias

    q = (q * softmax_scale).to(tl.float16)

    max_q = tl.max(q, 0)
    q_shift = max_q - tl.log(num_kv_pairs * 1.0)
    q = q - q_shift

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if SPLIT_K:
        k_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(num_kv_pairs, d_head),
            strides=(stride_km, stride_kd),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(num_kv_pairs, d_head),
            strides=(stride_vm, stride_vd),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
    else:
        k_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(num_kv_pairs, d_head),
            strides=(stride_km, stride_kd),
            offsets=(0, 0),
            block_shape=(num
