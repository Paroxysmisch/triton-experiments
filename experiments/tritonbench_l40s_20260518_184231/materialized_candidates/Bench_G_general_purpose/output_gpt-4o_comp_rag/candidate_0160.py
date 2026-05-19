import torch
import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k, v, h, s_k, s_v, s_h,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    # Define the program ID for indexing
    i_b, i_h, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Initialize pointers
    p_k = tl.make_block_ptr(k, (T, K), (s_k, 1), (i_t * BT, 0), (BT, BK))
    p_v = tl.make_block_ptr(v, (T, V), (s_v, 1), (i_t * BT, 0), (BT, BV))
    p_h = tl.make_block_ptr(h, (T, V), (s_h, 1), (i_t * BT, 0), (BT, BV))

    # Load and compute
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)
    b_h = tl.zeros([BT, V], dtype=tl.float32)

    # Compute the dot product for each block
    for _ in range(0, T, BT):
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (BT, 0))
        p_v = tl.advance(p_v, (BT, 0))

    # Store the result
    tl.store(p_h, b_h)

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q, k, h, o, s_q, s_k, s_h, s_o,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    # Define the program ID for indexing
    i_b, i_h, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Initialize pointers
    p_q = tl.make_block_ptr(q, (T, K), (s_q, 1), (i_t * BT, 0), (BT, BK))
    p_k = tl.make_block_ptr(k, (T, K), (s_k, 1), (i_t * BT, 0), (BT, BK))
    p_h = tl.make_block_ptr(h, (T, V), (s_h, 1), (i_t * BT, 0), (BT, BV))
    p_o = tl.make_block_ptr(o, (T, V), (s_o, 1), (i_t * BT, 0), (BT, BV))

    # Load and compute
    b_q = tl.load(p_q)
    b_k = tl.load(p_k)
    b_h = tl.load(p_h)
    b_o = tl.zeros([BT, V], dtype=tl.float32)

    # Compute the output
    for _ in range(0, T, BT):
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_o += tl.dot(b_s, b_h, allow_tf32=False)
        p_k = tl.advance(p_k, (BT, 0))
        p_h = tl.advance(p_h, (BT, 0))

    # Store the result
    tl.store(p_o, b_o)

@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    do, k, v, dh, s_do, s_k, s_v, s_dh,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    # Define the program ID for indexing
    i_b, i_h, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Initialize pointers
    p_do = tl.make_block_ptr(do, (T, V), (s_do, 1), (i_t * BT, 0), (BT, BV))
    p_k = tl.make_block_ptr(k, (T, K), (s_k, 1), (i_t * BT, 0), (BT, BK))
    p_v = tl.make_block_ptr(v, (T, V), (s_v, 1), (i_t * BT, 0), (BT, BV))
    p_dh = tl.make_block_ptr(dh, (T, V), (s_dh, 1), (i_t * BT, 0), (BT, BV))

    # Load and compute
    b_do = tl.load(p_do)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)
    b_dh = tl.zeros([BT, V], dtype=tl.float32)

    # Compute the gradient for h
    for _ in range(0, T, BT):
        b_dh += tl.dot(b_do, b_v, allow_tf32=False)
        p_v = tl.advance(p_v, (BT, 0))

    # Store the result
    tl.store(p_dh, b_dh)

@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    q, k, v, do, dq, dk, dv, s_q, s_k, s_v, s_do, s_dq, s_dk, s_dv,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    # Define the program ID for indexing
    i_b, i_h, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Initialize pointers
    p_q = tl.make_block_ptr(q, (T, K), (s_q, 1), (i_t * BT, 0), (BT, BK))
    p_k = tl.make_block_ptr(k, (T, K), (s_k, 1), (i_t * BT, 0), (BT, BK))
    p_v = tl.make_block_ptr(v, (T, V), (s_v, 1), (i_t * BT, 0), (BT, BV))
    p_do = tl.make_block_ptr(do, (T, V), (s_do, 1), (i_t * BT, 0), (BT, BV))
    p_dq = tl.make_block_ptr(dq, (T, K), (s_dq, 1), (i_t * BT, 0), (BT, BK))
    p_dk = tl.make_block_ptr(dk, (T, K), (s_dk, 1), (i_t * BT, 0), (BT, BK))
    p_dv = tl.make_block_ptr(dv, (T, V), (s_dv, 1), (i_t * BT, 0), (BT, BV))

    # Load and compute
    b_q = tl.load(p_q)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)
    b_do = tl.load(p_do)
    b_dq = tl.zeros([BT, K], dtype=tl.float32)
    b_dk = tl.zeros([BT, K], dtype=tl.float32)
    b_dv = tl.zeros([BT, V], dtype=tl.float32)

    # Compute the gradients for q, k, v
    for _ in range(0, T, BT):
        b_dq += tl.dot(b_do, b_k, allow_tf32=False)
        b_dk += tl.dot(b_q, b_do, allow_tf32=False)
        b_dv += tl.dot(b_do, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (BT, 0))
        p_v = tl.advance(p_v, (BT, 0))

    # Store the results
    tl.store(p_dq, b_dq)
    tl.store(p_dk, b_dk)
    tl.store(p_dv, b_dv)

class ChunkLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        BT, BK, BV = 128, 64, 64
        B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[3]
        
        # Allocate output tensors
        h = torch.empty(B, H, T, V, dtype=q.dtype, device=q.device)
        o = torch.empty(B, H, T, V, dtype=q.dtype, device=q.device)

        # Define grid size
        grid = (
