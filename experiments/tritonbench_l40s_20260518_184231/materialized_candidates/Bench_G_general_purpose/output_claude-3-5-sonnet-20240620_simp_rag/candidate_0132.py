import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q, k, v, o, z,  # pointers to tensors
    s_qk_h,         # stride for q/k head dimension
    s_qk_t,         # stride for q/k sequence dimension
    s_qk_d,         # stride for q/k feature dimension
    s_vo_h,         # stride for v/o head dimension
    s_vo_t,         # stride for v/o sequence dimension
    s_vo_d,         # stride for v/o feature dimension
    scale,          # scaling factor
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # num heads
    T: tl.constexpr,  # sequence length
    K: tl.constexpr,  # key dimension
    V: tl.constexpr,  # value dimension
    BTL: tl.constexpr,  # block size for sequence dimension
    BK: tl.constexpr,   # block size for key dimension
    BV: tl.constexpr,   # block size for value dimension
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BK)
    num_pid_v = tl.cdiv(V, BV)
    num_pid_m = tl.cdiv(T, BTL)
    
    # Initialize pointers to blocks
    offs_k = (pid % num_pid_k) * BK
    offs_v = ((pid // num_pid_k) % num_pid_v) * BV
    offs_m = ((pid // (num_pid_k * num_pid_v))) * BTL

    # Load q, k, v blocks
    q_ptr = tl.make_block_ptr(q, (T, K), (s_qk_t, s_qk_d), (offs_m, offs_k), (BTL, BK), (1, 0))
    k_ptr = tl.make_block_ptr(k, (K, T), (s_qk_d, s_qk_t), (offs_k, 0), (BK, T), (0, 1))
    v_ptr = tl.make_block_ptr(v, (T, V), (s_vo_t, s_vo_d), (0, offs_v), (T, BV), (1, 0))

    # Initialize accumulator
    b_q = tl.load(q_ptr, boundary_check=(0, 1))
    b_q = b_q * scale
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL, 1], dtype=tl.float32)

    # Main loop
    for i in range(0, T, BTL):
        # Load k, v blocks
        b_k = tl.load(k_ptr, boundary_check=(0, 1))
        b_v = tl.load(v_ptr, boundary_check=(0, 1))
        
        # Compute attention scores
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_p = tl.exp(b_s)
        
        # Update accumulator
        b_z += tl.sum(b_p, axis=1, keepdims=True)
        b_o += tl.dot(b_p, b_v, allow_tf32=False)
        
        # Advance pointers
        k_ptr = tl.advance(k_ptr, (0, BTL))
        v_ptr = tl.advance(v_ptr, (BTL, 0))

    # Store results
    o_ptr = tl.make_block_ptr(o, (T, V), (s_vo_t, s_vo_d), (offs_m, offs_v), (BTL, BV), (1, 0))
    z_ptr = tl.make_block_ptr(z, (T, 1), (s_vo_t, 1), (offs_m, 0), (BTL, 1), (1, 0))
    
    tl.store(o_ptr, b_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(z_ptr, b_z.to(z_ptr.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def parallel_rebased_bwd_kernel(
    q, k, v, do, dz, dq, dk, dv,  # pointers to tensors
    s_qk_h, s_qk_t, s_qk_d,       # strides for q/k
    s_vo_h, s_vo_t, s_vo_d,       # strides for v/o
    scale,                         # scaling factor
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BTL: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BK)
    num_pid_v = tl.cdiv(V, BV)
    num_pid_m = tl.cdiv(T, BTL)
    
    offs_k = (pid % num_pid_k) * BK
    offs_v = ((pid // num_pid_k) % num_pid_v) * BV
    offs_m = ((pid // (num_pid_k * num_pid_v))) * BTL

    # Load input gradients
    do_ptr = tl.make_block_ptr(do, (T, V), (s_vo_t, s_vo_d), (offs_m, offs_v), (BTL, BV), (1, 0))
    dz_ptr = tl.make_block_ptr(dz, (T, 1), (s_vo_t, 1), (offs_m, 0), (BTL, 1), (1, 0))
    
    b_do = tl.load(do_ptr, boundary_check=(0, 1))
    b_dz = tl.load(dz_ptr, boundary_check=(0, 1))

    # Initialize gradient accumulators
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    b_dk = tl.zeros([BK, T], dtype=tl.float32)
    b_dv = tl.zeros([T, BV], dtype=tl.float32)

    # Main backward loop
    for i in range(0, T, BTL):
        q_ptr = tl.make_block_ptr(q, (T, K), (s_qk_t, s_qk_d), (offs_m, offs_k), (BTL, BK), (1, 0))
        k_ptr = tl.make_block_ptr(k, (K, T), (s_qk_d, s_qk_t), (offs_k, i), (BK, BTL), (0, 1))
        v_ptr = tl.make_block_ptr(v, (T, V), (s_vo_t, s_vo_d), (i, offs_v), (BTL, BV), (1, 0))

        b_q = tl.load(q_ptr, boundary_check=(0, 1))
        b_k = tl.load(k_ptr, boundary_check=(0, 1))
        b_v = tl.load(v_ptr, boundary_check=(0, 1))

        # Compute attention pattern gradients
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_p = tl.exp(b_s)
        b_dp = b_do * b_dz

        # Accumulate gradients
        b_dq += tl.dot(b_dp, tl.trans(b_k), allow_tf32=False)
        b_dk += tl.dot(tl.trans(b_q), b_dp, allow_tf32=False)
        b_dv += tl.dot(tl.trans(b_p), b_do, allow_tf32=False)

    # Store gradients
    dq_ptr = tl.make_block_ptr(dq, (T, K), (s_qk_t, s_qk_d), (offs_m, offs_k), (BTL, BK), (1, 0))
    dk_ptr = tl.make_block_ptr(dk, (K, T), (s_qk_d, s_qk_t), (offs_k, 0), (BK, T), (0, 1))
    dv_ptr = tl.make_block_ptr(dv, (T, V), (s_vo_t, s_vo_d), (0, offs_v), (T, BV), (1, 0))

    tl.store(dq_ptr, b_dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dk_ptr, b_dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptr, b_dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))

class ParallelRebasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=None):
        # Setup dimensions and block sizes
        B, H, T, K = q.shape
        V = v.shape[-1]
        BTL = min(128, triton.next_power_of_2(T))
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
        
        scale = K ** -0.5 if scale is None else scale
        
        # Allocate output
        o = torch.empty((B, H, T, V), device=q.device, dtype=q.dtype)
        z = torch.empty((B, H, T, 1), device=q.device, dtype=q.dtype)
        
        # Launch kernel
        grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV) * triton.cdiv(T, BTL), B * H)
        parallel_rebased_fwd_kernel[grid](
            q, k, v, o, z,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            scale,
            B=B, H=H, T=T, K=K, V=V,
            BTL=BTL, BK=BK, BV=BV,
        )
        
        ctx.save_for_backward(q, k, v, z)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, z = ctx.saved_tensors
        scale = ctx.scale
        
        B, H, T, K = q.shape
        V = v.shape[-1]
        BTL = min(128, triton.next_power_of_2(T))
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
        
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV) * triton.cdiv(T, BTL), B * H)
        parallel_rebased_bwd_kernel[grid](
            q, k, v, do, z, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            scale,
            B=B, H=H, T=T, K=K, V=V,
            BTL=BTL, BK=BK, BV=BV,
        )
        
        return dq, dk, dv, None

def parallel_rebased(q, k, v, scale=None):
    """
    Parallel rebased attention mechanism
    
    Args:
        q: Query tensor of shape [B, H, T, K]
        k: Key tensor of shape [B, H, T, K] 
        v: Value tensor of shape [B, H, T, V]
        scale: Optional scaling factor (default: K^(-0.5))
    
    Returns:
        Output tensor of shape [B, H, T, V]
    """
    return ParallelRebasedFunction.apply(q, k, v, scale)
