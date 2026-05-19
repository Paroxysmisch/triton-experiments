import triton
import triton.language as tl

@triton.jit
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=4),
    ],
    key=['BT', 'BK', 'BV'],
)
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, h, g, do, dh, b_dq, b_dk, b_dg,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NH: tl.constexpr, NV: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(BT, BLOCK_SIZE_M)
    grid_n = tl.cdiv(BV, BLOCK_SIZE_N)
    grid_k = tl.cdiv(BK, BLOCK_SIZE_K)
    
    # Initialize gradients to zero
    b_dq[pid] = 0.0
    b_dk[pid] = 0.0
    b_dg[pid] = 0.0
    
    # Indices
    m = pid // (grid_n * grid_k)
    n = (pid // grid_k) % grid_n
    k_idx = pid % grid_k
    
    # Load data with boundary checks
    q_m = tl.load(q + m * BT * NH, mask=m < BT, other=0.0)
    k_n = tl.load(k + n * BK * NH, mask=n < BK, other=0.0)
    v_n = tl.load(v + n * BK * NV, mask=n < BK, other=0.0)
    h_n = tl.load(h + n * BK * NV, mask=n < BK, other=0.0)
    g_n = tl.load(g + n * BK * NH, mask=n < BK, other=0.0)
    do_n = tl.load(do + n * BK * NV, mask=n < BK, other=0.0)
    
    # Compute gradients
    for v_idx in range(BV):
        v_v = tl.load(v_n + v_idx * NV, mask=v_idx < NV, other=0.0)
        h_v = tl.load(h_n + v_idx * NV, mask=v_idx < NV, other=0.0)
        g_v = tl.load(g_n + v_idx * NH, mask=v_idx < NH, other=0.0)
        do_v = tl.load(do_n + v_idx * NV, mask=v_idx < NV, other=0.0)
        
        b_dq[pid] += do_v * h_v * g_v
        b_dk[pid] += do_v * h_v * g_v * q_m * k_n
        b_dg[pid] += do_v * h_v * q_m * k_n
    
    # Store gradients back to output tensors
    tl.store(b_dq + pid, b_dq[pid])
    tl.store(b_dk + pid, b_dk[pid])
    tl.store(b_dg + pid, b_dg[pid])
