import torch
import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    q, k, v, h, 
    s_q_h, s_q_t, s_q_d, 
    s_k_h, s_k_t, s_k_d, 
    s_v_h, s_v_t, s_v_d, 
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i_bh, i_t = tl.program_id(0), tl.program_id(1)
    # Pointers to blocks
    p_q = tl.make_block_ptr(q + i_bh * s_q_h, (T, K), (s_q_t, s_q_d), (i_t * BLOCK_SIZE, 0), (BLOCK_SIZE, K), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (0, 0), (T, K), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (0, 0), (T, V), (1, 0))
    
    # Load query, key, value blocks
    q_block = tl.load(p_q)
    k_block = tl.load(p_k)
    v_block = tl.load(p_v)

    # Compute dot product q * k^T and accumulate
    h_block = tl.zeros([BLOCK_SIZE, V], dtype=tl.float32)
    for t in range(0, T, BLOCK_SIZE):
        k_sub_block = k_block[:, t:t+BLOCK_SIZE]
        v_sub_block = v_block[t:t+BLOCK_SIZE, :]
        h_block += tl.dot(q_block, k_sub_block) @ v_sub_block

    # Store result
    p_h = tl.make_block_ptr(h + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BLOCK_SIZE, 0), (BLOCK_SIZE, V), (1, 0))
    tl.store(p_h, h_block)

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q, k, v, h, o, 
    s_q_h, s_q_t, s_q_d, 
    s_k_h, s_k_t, s_k_d, 
    s_v_h, s_v_t, s_v_d, 
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i_bh, i_t = tl.program_id(0), tl.program_id(1)
    # Pointers to blocks
    p_q = tl.make_block_ptr(q + i_bh * s_q_h, (T, K), (s_q_t, s_q_d), (i_t * BLOCK_SIZE, 0), (BLOCK_SIZE, K), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (0, 0), (T, K), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (0, 0), (T, V), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (0, 0), (T, V), (1, 0))

    # Load query, key, value, and intermediate h blocks
    q_block = tl.load(p_q)
    k_block = tl.load(p_k)
    v_block = tl.load(p_v)
    h_block = tl.load(p_h)

    # Compute dot product q * k^T and use h
    o_block = tl.zeros([BLOCK_SIZE, V], dtype=tl.float32)
    for t in range(0, T, BLOCK_SIZE):
        k_sub_block = k_block[:, t:t+BLOCK_SIZE]
        v_sub_block = v_block[t:t+BLOCK_SIZE, :]
        h_sub_block = h_block[t:t+BLOCK_SIZE, :]
        o_block += (tl.dot(q_block, k_sub_block) * h_sub_block) @ v_sub_block

    # Store result
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BLOCK_SIZE, 0), (BLOCK_SIZE, V), (1, 0))
    tl.store(p_o, o_block)

@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    q, k, v, do, dh, 
    s_q_h, s_q_t, s_q_d, 
    s_k_h, s_k_t, s_k_d, 
    s_v_h, s_v_t, s_v_d, 
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Similar implementation as forward, but computing dh
    # ...

@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    q, k, v, do, dq, dk, dv, 
    s_q_h, s_q_t, s_q_d, 
    s_k_h, s_k_t, s_k_d, 
    s_v_h, s_v_t, s_v_d, 
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Similar implementation as forward, but computing gradients for dq, dk, dv
    # ...

class ChunkLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        BLOCK_SIZE = 128
        B, H, T, K = q.shape
        V = v.shape[-1]
        o = torch.empty_like(v)

        grid = (B * H, T // BLOCK_SIZE)
        chunk_linear_attn_fwd_kernel_h[grid](
            q, k, v, o,
            q.stride(1), q.stride(2), q.stride(3),
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B=B, H=H, T=T, K=K, V=V,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(q, k, v)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        BLOCK_SIZE = 128
        B, H, T, K = q.shape
        V = v.shape[-1]

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid = (B * H, T // BLOCK_SIZE)
        chunk_linear_attn_bwd_kernel_dh[grid](
            q, k, v, do, dq,
            q.stride(1), q.stride(2), q.stride(3),
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B=B, H=H, T=T, K=K, V=V,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return dq, dk, dv

chunk_linear_attention = ChunkLinearAttentionFunction.apply
