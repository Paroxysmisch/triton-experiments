import torch
import triton
import triton.language as tl

# Define constants
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_SIZE_M = 8

@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_scale, k_scale,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_on,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q = tl.load(q_ptr + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk))
    k = tl.load(k_ptr + (offs_k[:, None] * stride_km + offs_n[None, :] * stride_kk))
    v = tl.load(v_ptr + (offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        q = tl.load(q_ptr + (offs_m[:, None] * stride_qm + (k + offs_k[None, :]) * stride_qk))
        k = tl.load(k_ptr + ((k + offs_k[:, None]) * stride_km + offs_n[None, :] * stride_kk))
        acc += tl.sum(q * k, axis=1) * q_scale * k_scale

    acc = tl.softmax(acc, axis=1)
    for k in range(0, K, BLOCK_K):
        v = tl.load(v_ptr + (offs_m[:, None] * stride_vm + (k + offs_k[None, :]) * stride_vk))
        acc = tl.sum(acc[:, None] * v, axis=1)

    tl.store(o_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on), acc)

@triton.jit
def _attn_fwd_inner(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_scale, k_scale,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_on,
    M, N, K,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q = tl.load(q_ptr + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk))
    k = tl.load(k_ptr + (offs_k[:, None] * stride_km + offs_n[None, :] * stride_kk))
    v = tl.load(v_ptr + (offs_m[:, None] * stride_vm + offs_k[None, :] * stride_vk))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        q = tl.load(q_ptr + (offs_m[:, None] * stride_qm + (k + offs_k[None, :]) * stride_qk))
        k = tl.load(k_ptr + ((k + offs_k[:, None]) * stride_km + offs_n[None, :] * stride_kk))
        acc += tl.sum(q * k, axis=1) * q_scale * k_scale

    if STAGE == 1:
        acc = tl.softmax(acc, axis=1)
    elif STAGE == 2:
        for k in range(0, K, BLOCK_K):
            v = tl.load(v_ptr + (offs_m[:, None] * stride_vm + (k + offs_k[None, :]) * stride_vk))
            acc = tl.sum(acc[:, None] * v, axis=1)

    tl.store(o_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on), acc)

def forward(q, k, v, q_scale, k_scale):
    M, K = q.shape
    N, _ = k.shape
    assert K == v.shape[1], "Key and Value dimensions must match"
    o = torch.empty((M, N), device=q.device, dtype=q.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _attn_fwd[grid](
        q, k, v, o,
        q_scale, k_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return o
