import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr,
    dq_ptr, dk_ptr, dg_ptr,
    M, N, K,
    stride_qm, stride_qk,
    stride_km, stride_kn,
    stride_vm, stride_vk,
    stride_gm, stride_gk,
    stride_hm, stride_hk,
    stride_dom, stride_don,
    stride_dqm, stride_dqk,
    stride_dkm, stride_dkn,
    stride_dgm, stride_dgk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create pointers for the current block
    q = tl.load(q_ptr + pid_m * BLOCK_SIZE_M * stride_qm + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_qk)
    k = tl.load(k_ptr + pid_n * BLOCK_SIZE_N * stride_kn + tl.arange(0, BLOCK_SIZE_N) * stride_km)
    v = tl.load(v_ptr + pid_n * BLOCK_SIZE_N * stride_vk + tl.arange(0, BLOCK_SIZE_N) * stride_vm)
    g = tl.load(g_ptr + pid_m * BLOCK_SIZE_M * stride_gm + tl.arange(0, BLOCK_SIZE_M) * stride_gk)
    h = tl.load(h_ptr + pid_m * BLOCK_SIZE_M * stride_hm + tl.arange(0, BLOCK_SIZE_M) * stride_hk)
    do = tl.load(do_ptr + pid_m * BLOCK_SIZE_M * stride_dom + tl.arange(0, BLOCK_SIZE_M) * stride_don)

    # Compute gradients
    b_dq = tl.zeros((BLOCK_SIZE_M, K), dtype=tl.float32)
    b_dk = tl.zeros((BLOCK_SIZE_N, K), dtype=tl.float32)
    b_dg = tl.zeros((BLOCK_SIZE_M, K), dtype=tl.float32)

    for k_idx in range(0, K, BLOCK_SIZE_K):
        # Load a block of the matrix
        q_block = tl.load(q_ptr + pid_m * BLOCK_SIZE_M * stride_qm + k_idx * stride_qk)
        k_block = tl.load(k_ptr + pid_n * BLOCK_SIZE_N * stride_kn + k_idx * stride_km)
        v_block = tl.load(v_ptr + pid_n * BLOCK_SIZE_N * stride_vk + k_idx * stride_vm)

        # Perform matrix multiplication and reductions
        b_dq += tl.dot(q_block, k_block)
        b_dk += tl.dot(k_block, v_block)
        b_dg += tl.dot(v_block, q_block)

    # Store results
    tl.store(dq_ptr + pid_m * BLOCK_SIZE_M * stride_dqm + tl.arange(0, BLOCK_SIZE_M) * stride_dqk, b_dq)
    tl.store(dk_ptr + pid_n * BLOCK_SIZE_N * stride_dkn + tl.arange(0, BLOCK_SIZE_N) * stride_dkm, b_dk)
    tl.store(dg_ptr + pid_m * BLOCK_SIZE_M * stride_dgm + tl.arange(0, BLOCK_SIZE_M) * stride_dgk, b_dg)


def chunk_bwd_dqkg_fn(q, k, v, g, h, do):
    # Extract shapes and strides
    M, K = q.shape
    _, N = k.shape

    # Allocate output tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)

    # Define grid dimensions
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 64))

    # Launch the Triton kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, g, h, do,
        dq, dk, dg,
        M, N, K,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        g.stride(0), g.stride(1),
        h.stride(0), h.stride(1),
        do.stride(0), do.stride(1),
        dq.stride(0), dq.stride(1),
        dk.stride(0), dk.stride(1),
        dg.stride(0), dg.stride(1)
    )

    return dq, dk, dg
