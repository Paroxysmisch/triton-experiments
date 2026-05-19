@triton.jit
def _triton_rope(
    q_ptr, k_ptr,
    cos, sin,
    head_dim      : tl.constexpr,
    backward      : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    pid    = tl.program_id(0)
    batch = pid // (head_dim // 2)
    pos_in_batch = pid % (head_dim // 2)
    pos_dim1 = batch * head_dim // 2
    pos_dim2 = (batch + 1) * head_dim // 2
    offset = tl.arange(0, BLOCK_SIZE)
    load_offset_q = (pos_dim1 + pos_in_batch + offset) * 2
    load_offset_k = (pos_dim2 + pos_in_batch + offset) * 2
    if backward:
        sin = -sin
    q = tl.load(q_ptr + load_offset_q, mask=offset < head_dim // 2, other=0).to(sin.dtype)
    k = tl.load(k_ptr + load_offset_k, mask=offset < head_dim // 2, other=0).to(sin.dtype)
    tl.store(q_ptr + load_offset_q, q * cos - k * sin, mask=offset < head_dim // 2)
    tl.store(k_ptr + load_offset_k, k * cos + q * sin, mask=offset < head_dim // 2)
