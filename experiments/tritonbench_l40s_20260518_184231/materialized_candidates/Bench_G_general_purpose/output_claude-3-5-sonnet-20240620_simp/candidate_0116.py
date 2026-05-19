import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'NUM_WARPS': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'NUM_WARPS': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr,
    dq_ptr, dk_ptr, dg_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_gm, stride_gk,
    stride_hm, stride_hk,
    stride_dom, stride_dok,
    stride_dqm, stride_dqk,
    stride_dkm, stride_dkk,
    stride_dgm, stride_dgk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Compute backward gradients for Q, K, and G in gated attention.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Current block
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offset pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulators
    b_dq = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    b_dk = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    b_dg = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Load blocks
    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    k = tl.load(k_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kk,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    v = tl.load(v_ptr + offs_m[:, None] * stride_vm + offs_n[None, :] * stride_vk,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    g = tl.load(g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gk,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    h = tl.load(h_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hk,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    do = tl.load(do_ptr + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_dok,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    
    # Compute gradients
    # dQ = dO * (K * G)
    b_dq = do * (k * g)
    # dK = dO * (Q * G)
    b_dk = do * (q * g)
    # dG = dO * (Q * K)
    b_dg = do * (q * k)
    
    # Write back results
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(dq_ptr + offs_m[:, None] * stride_dqm + offs_n[None, :] * stride_dqk,
             b_dq, mask=mask)
    tl.store(dk_ptr + offs_m[:, None] * stride_dkm + offs_n[None, :] * stride_dkk,
             b_dk, mask=mask)
    tl.store(dg_ptr + offs_m[:, None] * stride_dgm + offs_n[None, :] * stride_dgk,
             b_dg, mask=mask)

def chunk_bwd_dqkg_fn(q, k, v, g, h, do):
    """
    Wrapper function for the backward pass kernel.
    
    Args:
        q: Query tensor [M, K]
        k: Key tensor [M, K]
        v: Value tensor [M, K]
        g: Gate tensor [M, K]
        h: Hidden tensor [M, K]
        do: Gradient of output [M, K]
    
    Returns:
        dq: Gradient for query
        dk: Gradient for key
        dg: Gradient for gate
    """
    M, K = q.shape
    
    # Allocate output tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)
    
    # Define grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, g, h, do,
        dq, dk, dg,
        M, K, K,  # M, N, K
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        g.stride(0), g.stride(1),
        h.stride(0), h.stride(1),
        do.stride(0), do.stride(1),
        dq.stride(0), dq.stride(1),
        dk.stride(0), dk.stride(1),
        dg.stride(0), dg.stride(1),
    )
    
    return dq, dk, dg
