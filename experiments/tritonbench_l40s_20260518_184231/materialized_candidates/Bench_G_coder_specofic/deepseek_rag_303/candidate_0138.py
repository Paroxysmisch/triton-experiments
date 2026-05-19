import triton
from triton import language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    layout_csr_segment_start_indices,
    Out,
    PAST_SEQLEN: tl.constexpr,
    n_head: tl.constexpr,
    n_kv_head: tl.constexpr,
    blocksize_m: tl.constexpr,
    blocksize_n: tl.constexpr,
    blocksize_d: tl.constexpr,
    D: tl.constexpr,
    num_view: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    cur_q_batch = tl.program_id(0)
    cur_q_head = tl.program_id(1)
    pid_tz = tl.program_id(2)
    cur_k_head = pid_tz // (n_head // n_kv_head)
    cur_seg_head_start = pid_tz - cur_q_head * (n_head // n_kv_head)
    # initialize offset to the CSR layout
    cur_batch_in_all_start_index = tl.load(layout_csr_segment_start_indices + cur_q_batch)

    # get the start and end index of a segment
    seg_start_index, seg_end_index = 0, 0
    seg_idxes = cur_q_batch * n_head + cur_q_head
    if cur_seg_head_start == 0:
        seg_start_index_ = tl.load(layout_csr_row_indices + seg_idxes)
        seg_start_index = tl.maximum(seg_start_index_, 0)
    seg_end_index = tl.load(layout_csr_row_indices + seg_idxes + 1)
    seg_end_index = tl.minimum(seg_end_index, cur_batch_in_all_start_index)

    block_id = tl.math.div_euclidean(cur_q_head, n_head // n_kv_head)
    offs_m = tl.arange(0, blocksize_m)
    offs_n = tl.arange(0, blocksize_n)
    offs_d = tl.arange(0, blocksize_d)
    # 1 means in the middle of a block, 0 means before the block and -1 means after the block
    flag_in_block = tl.where(
        (offs_n[None, :] + PAST_SEQLEN) >= (seg_start_index + offs_m[:, None]), 1, 0
    )
    flag_in_block = tl.where(
        (seg_end_index + PAST_SEQLEN) > (offs_n[None, :] + PAST_SEQLEN), flag_in_block, -1
    )

    offs_q0 = (
        (seg_start_index + offs_m[:, None]) * D + (cur_q_head % n_kv_head) * blocksize_d
    )
    offs_k0 = (
        (block_id * blocksize_m + PAST_SEQLEN + offs_n[None, :]) * D
        + (cur_k_head % n_kv_head) * blocksize_d
    )

    acc = tl.zeros([blocksize_m, blocksize_d], dtype=tl.float32)
    denom = tl.zeros([blocksize_m], dtype=tl.float32) - float("inf")

    for start_c in range(0, NUM_D_BLOCKS):
        mask_d = (start_c * blocksize_d + offs_d[None, :]) < D
        q = tl.load(
            Q + offs_q0, mask=mask_d, other=0.0
        )  # TODO: this seems to be causing a clearer bug, check it out
        q = tl.dot(q, CASTFORMER_WEIGHTS[start_c], allow_tf32=False).to(
            Out.dtype.element_ty
        )
        offs_k = offs_k0 + start_c * blocksize_d * num_view
        flag_in_block1 = tl.where(seg_end_index > (offs_n + PAST_SEQLEN), 1, -1)
        k = tl.load(
            K + offs_k,
            mask=flag_in_block1[:, None] == 1 and mask_d,
            other=0.0,
        )
        qk = tl.dot(q, k, allow_tf32=False)

        mask = flag_in_block[:, None] == 1
        qk = tl.where(mask, qk, -float("inf"))
        qk = tl.where(mask, qk, 0.0)
        qk = tl.math.log1p(tl.math.exp(qk))
        # qk_m = tl.max(qk, 1)
        # qk_m = tl.maximum(qk_m, 0.0)
        # qk -= qk_m[:, None]
        # qk = tl.math.log1p(tl.math.exp(qk))

        denom = tl.maximum(denom, tl.max(qk, 1))

        qk = tl.exp(qk - denom[:, None])

        acc = acc * tl.exp(denom - (qk + tl.to(denom, qk.dtype))) + tl.dot(qk, V + offs_k)
        # acc += tl.dot(qk, V + offs_k)
        denom = tl.maximum(denom, tl.max(qk, 1))

        offs_q0 += blocksize_d
        offs_k0 += blocksize_d * num_view

    offset = tl.arange(0, out_size)
    output_ptrs = Out + offset + cur_q_batch * total_block_size + cur_q_head * blocksize_m * D
    mask = flag_in_block[:, None] > 0
    tl.store(output_ptrs, acc.to(Out.dtype.element_ty), mask=mask)

def query_key_value_head_block_sparse_attention_forward(
    q, k, v, layout_csr_row_indices, layout_csr_col_indices, layout_csr_segment_start_indices
):
    batch, head, seq_len = q.shape
    segment_len = layout_csr_row_indices.shape[1] - 1
    n_head = q.shape[1]
    assert q.shape == k.shape and q.shape == v.shape
    assert layout_csr_col_indices.shape[0] == segment_len * n_head
    layout_csr_block_id = (
        (layout_csr_col_indices[:, None] >= (layout_csr_row_indices[None, :] * n_head))
        & (layout_csr_col_indices[:, None] < (layout_csr_row_indices[None, :] + 1) * n_head)
    )
    block_id = triton.jit.filtered_reduction((batch, n_head), lambda x, y: x * n_head + y)(
        lambda x: layout_csr_block_id[x, 0] - 1, triton.jit.TILE
    )

    k = tl.trans(tl.reversible(k))
    q = tl.trans(q)
    batch_in_all_start_index = batch_in_all_start_index
    assert (
        layout_csr
