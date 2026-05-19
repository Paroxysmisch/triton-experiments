import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr, init_state_ptr, final_state_ptr,
    q_seq_len, k_seq_len, BT, B, H, D, S, scale,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    bid_h = bid * H + hid
    offset = bid_h * D + tl.arange(0, BLOCK_D)
    k_block_ptr = k_ptr + bid_h * S * D + offset
    v_block_ptr = v_ptr + bid_h * S * D + offset
    h_block_ptr = h_ptr + bid_h * BT * D + offset
    init_state_block_ptr = init_state_ptr + bid_h * D + offset
    final_state_block_ptr = final_state_ptr + bid_h * D + offset

    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if bid_h < B * H:
        if init_state_ptr != 0:
            h = tl.load(init_state_block_ptr, mask=offset < D, other=0.0)

    for s in range(0, k_seq_len, BLOCK_S):
        k = tl.load(k_block_ptr + s * D, mask=offset < D, other=0.0)
        v = tl.load(v_block_ptr + s * D, mask=offset < D, other=0.0)
        h = h + k * v * scale
        tl.store(h_block_ptr + s * D, h, mask=offset < D)

    if final_state_ptr != 0:
        tl.store(final_state_block_ptr, h, mask=offset < D)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, o_ptr, q_seq_len, k_seq_len, BT, B, H, D, S, scale,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    bid_h = bid * H + hid
    offset = bid_h * D + tl.arange(0, BLOCK_D)
    q_block_ptr = q_ptr + bid_h * S * D + offset
    k_block_ptr = k_ptr + bid_h * S * D + offset
    v_block_ptr = v_ptr + bid_h * S * D + offset
    o_block_ptr = o_ptr + bid_h * S * D + offset

    o = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if bid_h < B * H:
        for s in range(0, k_seq_len, BLOCK_S):
            k = tl.load(k_block_ptr + s * D, mask=offset < D, other=0.0)
            v = tl.load(v_block_ptr + s * D, mask=offset < D, other=0.0)
            q = tl.load(q_block_ptr + s * D, mask=offset < D, other=0.0)
            o = o + q * k * v * scale
            tl.store(o_block_ptr + s * D, o, mask=offset < D)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    q_ptr, do_ptr, dh_ptr, q_seq_len, k_seq_len, BT, B, H, D, S, scale,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    bid_h = bid * H + hid
    offset = bid_h * D + tl.arange(0, BLOCK_D)
    q_block_ptr = q_ptr + bid_h * S * D + offset
    do_block_ptr = do_ptr + bid_h * S * D + offset
    dh_block_ptr = dh_ptr + bid_h * BT * D + offset

    dh = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if bid_h < B * H:
        for s in range(0, k_seq_len, BLOCK_S):
            q = tl.load(q_block_ptr + s * D, mask=offset < D, other=0.0)
            do = tl.load(do_block_ptr + s * D, mask=offset < D, other=0.0)
            dh = dh + q * do * scale
            tl.store(dh_block_ptr + s * D, dh, mask=offset < D)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    do_ptr, dh_ptr, h_ptr, dq_ptr, dk_ptr, dv_ptr, q_seq_len, k_seq_len, BT, B, H, D, S, scale,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    bid_h = bid * H + hid
    offset = bid_h * D + tl.arange(0, BLOCK_D)
    do_block_ptr = do_ptr + bid_h * S * D + offset
    dh_block_ptr = dh_ptr + bid_h * BT * D + offset
    h_block_ptr = h_ptr + bid_h * BT * D + offset
    dq_block_ptr = dq_ptr + bid_h * S * D + offset
    dk_block_ptr = dk_ptr + bid_h * S * D + offset
    dv_block_ptr = dv_ptr + bid_h * S * D + offset

    dq = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dk = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dv = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if bid_h < B * H:
        for s in range(0, k_seq_len, BLOCK_S):
            do = tl.load(do_block_ptr + s * D, mask=offset < D, other=0.0)
            dh = tl.load(dh_block_ptr + s * D, mask=offset < D, other=0.0)
            h = tl.load(h_block_ptr + s * D, mask=offset < D, other=0.0)
            dq = dq + do * h * scale
            dk = dk + do * dh * scale
            dv = dv + do * dh * scale
            tl.store(dq_block_ptr + s * D, dq, mask=offset < D)
            tl.store(dk_block_ptr + s * D, dk, mask=offset < D)
            tl.store(dv_block_ptr + s * D, dv, mask=offset < D)

def chunk_fwd_h_fn(k, v, h, init_state=None, final_state=None, q_seq_len=0, k_seq_len=0, BT=0, B=0, H=0, D=0, S=0, scale=1.0):
    BLOCK_D = 128
    BLOCK_S = 128
    grid = (B, H)
    chunk_retention_fwd_kernel_h[grid](
        k, v, h, init_state, final_state,
        q_seq_len, k_seq_len, BT, B, H, D, S, scale,
        BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S
    )

def chunk_fwd_o_fn(q, k, v, o, q_seq_len=0, k_seq_len=0, BT=0, B=0, H=0, D=0, S=0, scale=1.0):
    BLOCK_D = 128
    BLOCK_S = 128
    grid = (B, H)
    chunk_retention_fwd_kernel_o[grid](
        q, k, v, o, q_seq_len, k_seq_len, BT, B, H, D, S, scale,
        BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S
    )

def chunk_bwd_dh_fn(q, do, dh, q_seq_len=0, k_seq_len=0, BT=0, B=0, H=0, D=0, S=0, scale=1.0):
    BLOCK_D = 128
    BLOCK_S = 128
    grid = (B, H)
    chunk_retention_bwd_kernel_dh[grid](
        q, do, dh, q_seq_len, k_seq_len, BT, B, H, D, S, scale,
        BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S
    )
