import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    initial_state,
    B,
    H,
    T,
    scale,
    NT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr = False,
    STORE_FINAL_STATE: tl.constexpr = False,
):
    # Kernel to calculate h, an intermediate tensor in the forward pass.
    b_index = tl.program_id(0)
    h_index = tl.program_id(1)
    tl.static_assert(NT % BLOCK == 0)
    n_block = NT // BLOCK
    # The buffer that stores h
    b_h = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)
    # The pointer to the first element of b_index's chunk
    p_k = k + b_index * T * HEAD_DIM + h_index * HEAD_DIM
    p_v = v + b_index * T * HEAD_DIM + h_index * HEAD_DIM
    if USE_INITIAL_STATE:
        # Load initial state into the buffer
        p_init_state = initial_state + b_index * H * HEAD_DIM + h_index * HEAD_DIM
        b_h = tl.load(p_init_state + tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    # Load k and v, update the buffer, and store the final state if needed
    for i in range(n_block):
        # The pointer to the first element of the i-th block
        block_offset = i * BLOCK
        p_ki = p_k + block_offset * HEAD_DIM
        p_vi = p_v + block_offset * HEAD_DIM
        # Element-wise kernel can be faster than matrix kernel
        # tl.store(p_h + block_offset * HEAD_DIM, b_h)
        # Use this when the compiler bug is fixed
        # b_h += tl.dot(tl.load(p_ki + tl.arange(0, BLOCK)[:, None] * HEAD_DIM),
        #               tl.load(p_vi + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        # Use this when the bug is fixed
        # b_h = b_h + tl.dot(tl.load(p_ki[:, None] + tl.arange(0, BLOCK)[:, None] * HEAD_DIM),
        #                    tl.load(p_vi[None, :] + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        # A workaround
        b_h = b_h + tl.dot(tl.load(p_ki + tl.arange(0, BLOCK)[:, None] * HEAD_DIM)[:, None],
                           tl.load(p_vi + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        if i == n_block - 1 and STORE_FINAL_STATE:
            p_final_state = initial_state + b_index * H * HEAD_DIM + h_index * HEAD_DIM + (NT - 1) * HEAD_DIM
            tl.store(p_final_state + tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :],
                     b_h)
        b_h = b_h * tl.math.exp2(chunk_decay_func(i * BLOCK / T))[:, None]
    tl.store((tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]) + \
             (h_index * HEAD_DIM + b_index * T * HEAD_DIM), b_h)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q,
    k,
    v,
    h,
    initial_state,
    B,
    H,
    T,
    scale,
    NT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr = False,
):
    # Kernel to calculate o, the output tensor in the forward pass.
    b_index = tl.program_id(0)
    head_index = tl.program_id(1)
    tl.static_assert(NT % BLOCK == 0)
    n_block = NT // BLOCK
    # The buffer that stores intermediate results
    b_o = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)
    b_s = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)
    # The pointer to the first element of b_index's chunk
    p_q = q + b_index * T * HEAD_DIM + head_index * HEAD_DIM
    p_k = k + b_index * T * HEAD_DIM + head_index * HEAD_DIM
    p_v = v + b_index * T * HEAD_DIM + head_index * HEAD_DIM
    # Load initial state into the buffer
    if USE_INITIAL_STATE:
        p_init_state = initial_state + b_index * H * HEAD_DIM + head_index * HEAD_DIM
        b_s = tl.load(p_init_state + tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    # q * (exp(x) * v + exp(y) * h)
    for i in range(n_block):
        block_offset = i * BLOCK
        # The pointer to the first element of the i-th block
        p_qi = p_q + block_offset * HEAD_DIM
        p_hi = h + block_offset + b_index * T * HEAD_DIM + head_index * HEAD_DIM
        p_ki = p_k + block_offset * HEAD_DIM
        p_vi = p_v + block_offset * HEAD_DIM
        # Use this when the compiler bug is fixed
        # b_s += tl.dot(tl.load(p_ki + tl.arange(0, BLOCK)[:, None] * HEAD_DIM),
        #               tl.load(p_hi + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        # Use this when the bug is fixed
        # b_o += tl.dot(tl.load(p_qi + tl.arange(0, BLOCK)[:, None] * HEAD_DIM),
        #               tl.load(p_vi + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        # A workaround
        b_s = b_s + tl.dot(tl.load(p_ki + tl.arange(0, BLOCK)[:, None] * HEAD_DIM)[:, None],
                           tl.load(p_hi + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        b_o = b_o + tl.dot(tl.load(p_qi + tl.arange(0, BLOCK)[:, None] * HEAD_DIM)[:, None],
                           tl.load(p_vi + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        # The compiler has a bug. It cannot optimize the code below.
        # b_o = b_o + tl.dot(tl.load(p_qi[:, None] + tl.arange(0, BLOCK)[:, None] * HEAD_DIM),
        #                    tl.load(p_vi[None, :] + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        # b_s = b_s + tl.dot(tl.load(p_ki[:, None] + tl.arange(0, BLOCK)[:, None] * HEAD_DIM),
        #                    tl.load(p_hi[None, :] + tl.arange(0, BLOCK)[None, :]), allow_tf32=True)
        # Scale the result
        b_o = b_o * tl.math.exp2(chunk_decay_func(i * BLOCK / T) - 1)[:, None]
        b_s = b_s * tl.math.exp2(chunk_decay_func(i * BLOCK / T))[:, None]
        # Store the result
        tl.store((tl.arange(0, BLOCK)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]) + \
                 (head_index * HEAD_DIM + b_index * T * HEAD_DIM + block_offset), b_o + b_s)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    q,
    dk,
    dv,
    h,
    dh,
    B,
    H,
    T,
    NT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Kernel to calculate dh in the backward pass.
    b_index = tl.program_id(0)
    head_index = tl.program_id(1)
    tl.static_assert(NT % BLOCK == 0)
    n_block = NT // BLOCK
    # The pointer to the first element of the i-th block
    p_q = q + b_index * T * HEAD_DIM + head_index * HEAD_DIM
    p_hi = h + b_index * T * HEAD_DIM + head_index * HEAD_DIM
    p_dh = dh + b_index * T * HEAD_DIM + head_index * HEAD_DIM
    p_dk = dk + b_index * T * HEAD_DIM + head_index * HEAD_DIM
    # Load dh
    b_dh = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)
    b_ds = tl.zeros([BLOCK, HEAD_DIM], dtype=tl.float32)
    for i in range(n_block - 1, -1, -1):
        block_offset = i * BLOCK
        # The pointer to the first element of the i-th block
        p_qi = p_q + block_offset * HEAD_DIM
        p_dhi = p_hi + block_offset * HEAD_DIM
        p_dqi = p_qi
        p
