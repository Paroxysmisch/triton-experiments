import torch
import triton
import triton.language as tl

# Forward Kernel for Inter-SubBlock A Computation
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q_ptr, k_ptr, g_ptr, A_ptr,
    B, H, M, N, D,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_gb, stride_gh, stride_gm, stride_gn,
    stride_Ab, stride_Ah, stride_Am, stride_An,
    scale,
    BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)

    # Skip upper triangular blocks
    if pid_m <= pid_n:
        return

    # Offsets for block processing
    off_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    off_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    off_d = tl.arange(0, D_HEAD)

    # Create block pointers with masking
    q_offs = pid_b * stride_qb + pid_h * stride_qh + off_m[:, None] * stride_qm + off_d[None, :] * stride_qd
    k_offs = pid_b * stride_kb + pid_h * stride_kh + off_n[:, None] * stride_km + off_d[None, :] * stride_kd
    g_offs = pid_b * stride_gb + pid_h * stride_gh + off_m[:, None] * stride_gm + off_n[None, :] * stride_gn

    # Mask for valid elements
    mask_q = (off_m[:, None] < M) & (off_d[None, :] < D)
    mask_k = (off_n[:, None] < N) & (off_d[None, :] < D)
    mask_g = (off_m[:, None] < M) & (off_n[None, :] < N)

    # Load blocks with masking
    q = tl.load(q_ptr + q_offs, mask=mask_q, other=0.0)
    k = tl.load(k_ptr + k_offs, mask=mask_k, other=0.0)
    g = tl.load(g_ptr + g_offs, mask=mask_g, other=0.0)

    # Compute attention scores
    scores = tl.dot(q, k, trans_b=True) * scale
    scores = tl.exp(scores) * g

    # Store results
    a_offs = pid_b * stride_Ab + pid_h * stride_Ah + off_m[:, None] * stride_Am + off_n[None, :] * stride_An
    tl.store(A_ptr + a_offs, scores, mask=mask_g)

# Forward Kernel for Intra-SubBlock A Computation
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q_ptr, k_ptr, g_ptr, A_ptr,
    B, H, M, D,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_gb, stride_gh, stride_gm,
    stride_Ab, stride_Ah, stride_Am,
    scale,
    BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    off_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    off_d = tl.arange(0, D_HEAD)

    # Load diagonal block
    q_offs = pid_b * stride_qb + pid_h * stride_qh + off_m[:, None] * stride_qm + off_d[None, :] * stride_qd
    k_offs = pid_b * stride_kb + pid_h * stride_kh + off_m[:, None] * stride_km + off_d[None, :] * stride_kd
    g_offs = pid_b * stride_gb + pid_h * stride_gh + off_m[:, None] * stride_gm + off_m[None, :] * stride_gm

    mask = (off_m[:, None] < M) & (off_d[None, :] < D)
    mask_g = (off_m[:, None] < M) & (off_m[None, :] < M)

    q = tl.load(q_ptr + q_offs, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + g_offs, mask=mask_g, other=0.0)

    # Compute diagonal scores
    scores = tl.dot(q, k, trans_b=True) * scale
    scores = tl.exp(scores) * g

    a_offs = pid_b * stride_Ab + pid_h * stride_Ah + off_m[:, None] * stride_Am + off_m[None, :] * stride_Am
    tl.store(A_ptr + a_offs, scores, mask=mask_g)

# Wrapper Function for A Computation
def chunk_fwd_intra_gated_gk_fn(q, k, g, A, scale):
    B, H, M, D = q.shape
    N = k.shape[2]
    BLOCK = 64
    D_HEAD = D
    
    grid = (B, H, triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    
    # Run inter-subblock kernel
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q, k, g, A,
        B, H, M, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        scale,
        BLOCK=BLOCK,
        D_HEAD=D_HEAD,
    )
    
    # Run intra-subblock kernel
    grid_intra = (B, H, triton.cdiv(M, BLOCK))
    chunk_gla_fwd_A_kernel_intra_sub_intra[grid_intra](
        q, k, g, A,
        B, H, M, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        A.stride(0), A.stride(1), A.stride(2),
        scale,
        BLOCK=BLOCK,
        D_HEAD=D_HEAD,
    )

# Forward Kernel for Output Computation
@triton.jit
def chunk_gla_fwd_kernel_o(
    A_ptr, v_ptr, g_ptr, o_ptr,
    B, H, M, D,
    stride_Ab, stride_Ah, stride_Am, stride_An,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_gb, stride_gh, stride_gm,
    stride_ob, stride_oh, stride_om, stride_od,
    BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    off_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    off_d = tl.arange(0, D_HEAD)

    # Load attention matrix block
    a_offs = pid_b * stride_Ab + pid_h * stride_Ah + off_m[:, None] * stride_Am + off_m[None, :] * stride_An
    v_offs = pid_b * stride_vb + pid_h * stride_vh + off_m[:, None] * stride_vm + off_d[None, :] * stride_vd
    g_offs = pid_b * stride_gb + pid_h * stride_gh + off_m[:, None] * stride_gm

    mask = (off_m[:, None] < M) & (off_d[None, :] < D)
    mask_g = (off_m[:, None] < M)

    A = tl.load(A_ptr + a_offs, mask=mask_g, other=0.0)
    v = tl.load(v_ptr + v_offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + g_offs, mask=mask_g, other=0.0)

    # Compute output block
    o = tl.dot(A, v) * g[:, None]

    # Store output
    o_offs = pid_b * stride_ob + pid_h * stride_oh + off_m[:, None] * stride_om + off_d[None, :] * stride_od
    tl.store(o_ptr + o_offs, o, mask=mask)

# Wrapper Function for Output Computation
def chunk_fwd_o_gated_gk_fn(A, v, g, o):
    B, H, M, D = o.shape
    BLOCK = 64
    D_HEAD = D
    
    grid = (B, H, triton.cdiv(M, BLOCK))
    
    chunk_gla_fwd_kernel_o[grid](
        A, v, g, o,
        B, H, M, D,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BLOCK=BLOCK,
        D_HEAD=D_HEAD,
    )
