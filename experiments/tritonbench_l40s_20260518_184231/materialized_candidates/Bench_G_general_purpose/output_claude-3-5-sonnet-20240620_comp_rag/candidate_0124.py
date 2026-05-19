import torch
import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    q, k, v, o,  # Main tensors [B,H,L,K/V] 
    s_qk_h, s_qk_t, s_qk_d,  # Strides for q/k
    s_vo_h, s_vo_t, s_vo_d,  # Strides for v/o
    scale,  # Scale factor K**-0.5
    B: tl.constexpr, H: tl.constexpr,  # Batch size, num heads
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,  # Sequence length, key/value dims
    BTL: tl.constexpr, BTS: tl.constexpr,  # Block sizes for sequences
    BK: tl.constexpr, BV: tl.constexpr,  # Block sizes for K/V dims
):
    # Get program IDs for parallelization
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H

    # Calculate decay rate based on head index
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    # Create block pointers for efficient memory access
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), 
                           (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                           (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                           (0, i_v * BV), (BTS, BV), (1, 0))

    # Load query block and apply scaling
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    # Process non-overlapping blocks
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_h[None, :]
        b_o = b_o * tl.math.exp2(b_b * BTS)
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # Process overlapping blocks
    o_q = tl.arange(0, BTL)
    d_q = tl.math.exp2(tl.arange(0, BTL) * b_b)
    b_o *= d_q[:, None]

    # Store final output
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V),
                           (s_vo_t, s_vo_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

class ParallelRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Configure block sizes and launch parameters
        BTL, BTS = 128, 32
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        
        # Launch kernel with appropriate grid
        grid = (NK * NV, triton.cdiv(T, BTL), B * H)
        o = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)
        parallel_retention_fwd_kernel[grid](...)
        return o.sum(0)

    @staticmethod
    def backward(ctx, do):
        # Similar configuration for backward pass
        dq, dk, dv = torch.empty(...), torch.empty(...), torch.empty(...)
        parallel_retention_bwd_kernel[grid](...)
        return dq.sum(0), dk.sum(0), dv.sum(0)
