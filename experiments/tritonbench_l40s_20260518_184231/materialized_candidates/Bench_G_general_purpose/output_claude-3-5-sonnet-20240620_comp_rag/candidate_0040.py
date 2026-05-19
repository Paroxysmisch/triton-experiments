import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _score_kernel(
    Q, K, M, Out,
    stride_qm, stride_qk,
    stride_km, stride_kn,
    stride_mm, stride_mn,
    stride_om, stride_on,
    M, N, K,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_in_group
    group_size_m = min(num_pid_m - first_pid_m, num_pid_in_group)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (offs_k[:, None] * stride_km + offs_n[None, :] * stride_kn)
    m_ptrs = M + (offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        accumulator += qk
        q_ptrs += BLOCK_K * stride_qk
        k_ptrs += BLOCK_K * stride_km
    
    accumulator = accumulator * sm_scale
    
    if SLIDING_WINDOW:
        m_mask = (offs_m[:, None] >= (offs_n[None, :] - WINDOW_SIZE)) & (offs_m[:, None] <= (offs_n[None, :] + WINDOW_SIZE))
    else:
        m_mask = offs_m[:, None] < N
    
    if CAUSAL:
        m_mask = m_mask & (offs_m[:, None] >= offs_n[None, :])
    
    mask = tl.load(m_ptrs, mask=m_mask)
    output = accumulator + mask
    
    o_ptrs = Out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(o_ptrs, output, mask=m_mask)

def get_score(q, k, m, causal=False, sm_scale=None, sliding_window=False, window_size=None):
    BLOCK_M, BLOCK_N = 128, 128
    
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk, f"Dimensions of q and k must match, but got {Lq} and {Lk}"
    
    batch_size_q, n_heads_q, n_ctx_q, d_head_q = q.shape
    batch_size_k, n_heads_k, n_ctx_k, d_head_k = k.shape
    
    assert batch_size_q == batch_size_k and n_heads_q == n_heads_k, f"Batch size and number of heads must match for q and k"
    
    M, N, K = n_ctx_q, n_ctx_k, d_head_q
    
    if sm_scale is None:
        sm_scale = 1.0 / (K ** 0.5)
    
    out = torch.empty((batch_size_q, n_heads_q, n_ctx_q, n_ctx_k), device=q.device, dtype=q.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    def run_kernel():
        _score_kernel[grid](
            q, k, m, out,
            q.stride(2), q.stride(3),
            k.stride(2), k.stride(3),
            m.stride(0), m.stride(1),
            out.stride(2), out.stride(3),
            M, N, K,
            sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
            CAUSAL=causal,
            SLIDING_WINDOW=sliding_window,
            WINDOW_SIZE=window_size if sliding_window else 0,
        )
    
    try:
        run_kernel()
    except triton.OutOfResources:
        BLOCK_M, BLOCK_N = BLOCK_M // 2, BLOCK_N // 2
        run_kernel()
    
    return out

# Example usage
# q = torch.randn(1, 12, 1024, 64, device='cuda')
# k = torch.randn(1, 12, 1024, 64, device='cuda')
# m = torch.zeros(1, 1024, 1024, device='cuda')
# output = get_score(q, k, m, causal=True, sliding_window=True, window_size=256)
