import torch
import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr,
    T, D,
    stride_kt, stride_kd,
    stride_vt, stride_vd,
    stride_ht, stride_hd,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute time index
    t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create block pointers
    k_block_ptr = tl.make_block_ptr(
        k_ptr, (T, D),
        (stride_kt, stride_kd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr, (T, D),
        (stride_vt, stride_vd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    
    # Load k and v blocks
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    
    # Compute h = k @ v.T
    h = tl.dot(k, tl.trans(v))
    
    # Store result
    h_block_ptr = tl.make_block_ptr(
        h_ptr, (T, T),
        (stride_ht, stride_ht),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, T)
    )
    tl.store(h_block_ptr, h)

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q_ptr, h_ptr, o_ptr,
    T, D,
    stride_qt, stride_qd,
    stride_ht, stride_hd,
    stride_ot, stride_od,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Create block pointers
    q_block_ptr = tl.make_block_ptr(
        q_ptr, (T, D),
        (stride_qt, stride_qd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    h_block_ptr = tl.make_block_ptr(
        h_ptr, (T, T),
        (stride_ht, stride_ht),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, T)
    )
    
    # Load blocks
    q = tl.load(q_block_ptr)
    h = tl.load(h_block_ptr)
    
    # Compute o = h @ q
    o = tl.dot(h, q)
    
    # Store result
    o_block_ptr = tl.make_block_ptr(
        o_ptr, (T, D),
        (stride_ot, stride_od),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    tl.store(o_block_ptr, o)

@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    q_ptr, do_ptr, dh_ptr,
    T, D,
    stride_qt, stride_qd,
    stride_dot, stride_dod,
    stride_dht, stride_dhd,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Create block pointers
    q_block_ptr = tl.make_block_ptr(
        q_ptr, (T, D),
        (stride_qt, stride_qd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    do_block_ptr = tl.make_block_ptr(
        do_ptr, (T, D),
        (stride_dot, stride_dod),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    
    # Load blocks
    q = tl.load(q_block_ptr)
    do = tl.load(do_block_ptr)
    
    # Compute dh = do @ q.T
    dh = tl.dot(do, tl.trans(q))
    
    # Store result
    dh_block_ptr = tl.make_block_ptr(
        dh_ptr, (T, T),
        (stride_dht, stride_dhd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, T)
    )
    tl.store(dh_block_ptr, dh)

@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    q_ptr, k_ptr, v_ptr, h_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    T, D,
    stride_qt, stride_qd,
    stride_kt, stride_kd,
    stride_vt, stride_vd,
    stride_ht, stride_hd,
    stride_dot, stride_dod,
    stride_dqt, stride_dqd,
    stride_dkt, stride_dkd,
    stride_dvt, stride_dvd,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Create block pointers for all tensors
    q_block_ptr = tl.make_block_ptr(
        q_ptr, (T, D),
        (stride_qt, stride_qd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr, (T, D),
        (stride_kt, stride_kd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr, (T, D),
        (stride_vt, stride_vd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    h_block_ptr = tl.make_block_ptr(
        h_ptr, (T, T),
        (stride_ht, stride_hd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, T)
    )
    do_block_ptr = tl.make_block_ptr(
        do_ptr, (T, D),
        (stride_dot, stride_dod),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    
    # Load blocks
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    h = tl.load(h_block_ptr)
    do = tl.load(do_block_ptr)
    
    # Compute gradients
    dq = tl.dot(h, do)
    dk = tl.dot(do, tl.trans(v))
    dv = tl.dot(tl.trans(k), do)
    
    # Store results
    dq_block_ptr = tl.make_block_ptr(
        dq_ptr, (T, D),
        (stride_dqt, stride_dqd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    dk_block_ptr = tl.make_block_ptr(
        dk_ptr, (T, D),
        (stride_dkt, stride_dkd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    dv_block_ptr = tl.make_block_ptr(
        dv_ptr, (T, D),
        (stride_dvt, stride_dvd),
        (pid * BLOCK_SIZE, 0),
        (BLOCK_SIZE, D)
    )
    
    tl.store(dq_block_ptr, dq)
    tl.store(dk_block_ptr, dk)
    tl.store(dv_block_ptr, dv)

class ChunkLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        T, D = q.shape
        device = q.device
        
        # Allocate output tensors
        h = torch.empty((T, T), device=device)
        o = torch.empty((T, D), device=device)
        
        # Configure grid and block sizes
        BLOCK_SIZE = 32
        grid = lambda meta: (triton.cdiv(T, BLOCK_SIZE),)
        
        # Forward pass kernels
        chunk_linear_attn_fwd_kernel_h[grid](
            k, v, h,
            T, D,
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            h.stride(0), h.stride(1),
            BLOCK_SIZE
        )
        
        chunk_linear_attn_fwd_kernel_o[grid](
            q, h, o,
            T, D,
            q.stride(0), q.stride(1),
            h.stride(0), h.stride(1),
            o.stride(0), o.stride(1),
            BLOCK_SIZE
        )
        
        ctx.save_for_backward(q, k, v, h)
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, h = ctx.saved_tensors
        T, D = q.shape
        device = q.device
        
        # Allocate gradient tensors
        dh = torch.empty((T, T), device=device)
        dq = torch.empty((T, D), device=device)
        dk = torch.empty((T, D), device=device)
        dv = torch.empty((T, D), device=device)
        
        # Configure grid
        BLOCK_SIZE = 32
        grid = lambda meta: (triton.cdiv(T, BLOCK_SIZE),)
        
        # Backward pass kernels
        chunk_linear_attn_bwd_kernel_dh[grid](
            q, grad_output, dh,
            T, D,
            q.stride(0), q.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            dh.stride(0), dh.stride(1),
            BLOCK_SIZE
        )
        
        chunk_linear_attn_bwd_kernel_dqkv[grid](
            q, k, v, h, grad_output,
            dq, dk, dv,
            T, D,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            h.stride(0), h.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            dq.stride(0), dq.stride(1),
            dk.stride(0), dk.stride(1),
            dv.stride(0), dv.stride(1),
            BLOCK_SIZE
        )
        
        return dq, dk, dv

def chunk_linear_attention(q, k, v):
    return ChunkLinearAttentionFunction.apply(q, k, v)
