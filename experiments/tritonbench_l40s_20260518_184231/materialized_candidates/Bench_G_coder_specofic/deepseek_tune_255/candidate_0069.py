import torch
import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, sm_scale,  # input tensors
    block_count, block_offset, column_count, column_index,
    seqlens,  # sequence length
    Out,  # output tensors
    max_block_count, max_column_count,  # dimension
    qk_scale,  # scalar
    BLOCK_SIZE: tl.constexpr,
):
    # kernel function to compute mixed sparse attention
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_len = tl.load(seqlens + cur_batch)
    cur_block_count = tl.load(block_count + cur_batch)
    cur_column_count = tl.load(column_count + cur_batch)
    if cur_block_count == 0 or cur_column_count == 0:
        return
    cur_block_count = tl.minimum(cur_block_count, max_block_count)
    cur_column_count = tl.minimum(cur_column_count, max_column_count)

    cur_block_offset = tl.load(block_offset + cur_batch)
    cur_column_index = tl.load(column_index + cur_batch)

    start_col_index = cur_column_index
    block_end = cur_block_offset + cur_block_count
    column_end = cur_column_index + cur_column_count

    # initialize offsets
    offs_n = cur_batch * seq_len * BLOCK_SIZE
    offs_d = cur_head * BLOCK_SIZE
    offs_b = cur_block_offset * BLOCK_SIZE
    off_q = (offs_n + offs_d + offs_b)[None, :]
    off_k = (offs_d + offs_b)[:, None]

    # initialize pointer to m and l
    m_ptr = tl.make_block_ptr(
        base=Q + off_q,
        shape=(BLOCK_SIZE, BLOCK_SIZE),
        strides=(1, BLOCK_SIZE),
        offsets=(0, 0),
        block_shape=(cur_block_count, BLOCK_SIZE),
        order=(0, 1),
    )
    l_ptr = tl.make_block_ptr(
        base=K + off_k,
        shape=(BLOCK_SIZE, BLOCK_SIZE),
        strides=(BLOCK_SIZE, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, cur_column_count),
        order=(1, 0),
    )

    # load Q
    q = tl.load(Q + off_q, boundary_check=(0, 1))

    # apply qk_scale
    if qk_scale is not None:
        q = q * qk_scale

    # compute max_x and max_y
    max_x = tl.full([cur_block_count, 1], value=-float("inf"), dtype=tl.float32)
    max_y = tl.full([1, cur_column_count], value=-float("inf"), dtype=tl.float32)

    # load K and compute x
    x = tl.load(K + off_k, boundary_check=(0, 1))
    x = tl.where(block_end > (start_col_index + BLOCK_SIZE), x, float("-inf"))
    x = tl.where(column_end > (start_col_index + BLOCK_SIZE), x, float("-inf"))
    x = tl.maximum(x, max_x)
    max_x = tl.maximum(tl.max(x, axis=1)[:, None], max_x)

    # compute z
    z = max_x + tl.log(tl.sum(tl.exp(x - max_x), axis=1))[:, None]

    # compute numerator
    numerator = tl.exp(x - z)

    # load V
    v = tl.load(V + off_k, boundary_check=(0, 1))
    v = tl.where(block_end > (start_col_index + BLOCK_SIZE), v, 0)
    v = tl.where(column_end > (start_col_index + BLOCK_SIZE), v, 0)
    # compute denominator
    denominator = tl.sum(numerator, axis=1)[:, None]

    # compute final output
    m = numerator / denominator
    l = (v * numerator[:, :, None]).sum(axis=1) / denominator

    # compute final output
    l = l + tl.load(K + off_k, boundary_check=(0, 1)) * sm_scale
    m = tl.where(block_end > (start_col_index + BLOCK_SIZE), m, 0)
    m = tl.where(column_end > (start_col_index + BLOCK_SIZE), m, 0)

    l_ptr = tl.make_block_ptr(
        base=l,
        shape=(BLOCK_SIZE, BLOCK_SIZE),
        strides=(1, BLOCK_SIZE),
        offsets=(0, 0),
        block_shape=(cur_block_count, BLOCK_SIZE),
        order=(0, 1),
    )
    m_ptr = tl.make_block_ptr(
        base=m,
        shape=(BLOCK_SIZE, BLOCK_SIZE),
        strides=(1, BLOCK_SIZE),
        offsets=(0, 0),
        block_shape=(cur_block_count, BLOCK_SIZE),
        order=(0, 1),
    )

    # compute final output
    final_output = tl.zeros([cur_block_count, BLOCK_SIZE], dtype=tl.float32)
    for _ in range(0, cur_column_count, BLOCK_SIZE):
        final_output += tl.load(l_ptr, boundary_check=(0, 1)) * tl.load(
            m_ptr, boundary_check=(0, 1)
        )
        tl.store(l_ptr, value=0, boundary_check=(0, 1))
        tl.store(m_ptr, value=0, boundary_check=(0, 1))
        l_ptr = tl.advance(l_ptr, [0, BLOCK_SIZE])
        m_ptr = tl.advance(m_ptr, [0, BLOCK_SIZE])

    # store output
    tl.store(Out + off_q, final_output.to(Q.dtype.element_ty), boundary_check=(0, 1))


def _triton_mixed_sparse_attention(q, k, v, block_count, block_offset, column_index, column_count, seqlens, sm_scale, qk_scale=None):
    # wrapper function to compute mixed sparse attention
    BLOCK_SIZE = 64
    # initialize output tensor
    out = torch.empty_like(q)
    batch, head = q.shape[:2]
    max_block_count = (
        torch.cumsum(seqlens, axis=0) - seqlens
    )  # compute max_block_count
    max_block_count = max_block_count.detach().to(torch.int32).tolist()
    max_column_count = column_count.detach().to(torch.int32).tolist()

    grid = (batch, head)
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        block_count,
        block_offset,
        column_count,
        column_index,
        seqlens,
        out,
        max_block
