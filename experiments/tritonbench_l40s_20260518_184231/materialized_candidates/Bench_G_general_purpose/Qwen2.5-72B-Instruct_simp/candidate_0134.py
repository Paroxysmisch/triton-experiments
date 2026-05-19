import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_on,
    Lk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(Q.shape[0], BLOCK_M)
    num_pid_n = tl.cdiv(V.shape[1], BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) // num_pid_m

    # Compute the offsets for Q, K, V
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_km)
    v_ptrs = V + (offs_m[:, None] * stride_vm + offs_n[None, :] * stride_vk)

    # Initialize the output and scale
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = 1.0 / (Lk ** 0.5)

    # Compute the dot product
    for k in range(0, Lk, BLOCK_DMODEL):
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        acc += tl.dot(q, k, allow_tf32=True) * scale
        q_ptrs += BLOCK_DMODEL * stride_qk
        k_ptrs += BLOCK_DMODEL * stride_km

    # Compute the softmax
    acc = tl.softmax(acc, axis=1)

    # Compute the output
    out_ptrs = Out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    for k in range(0, Lk, BLOCK_DMODEL):
        v = tl.load(v_ptrs)
        acc = tl.dot(acc, v, allow_tf32=True)
        v_ptrs += BLOCK_DMODEL * stride_vk

    # Store the result
    tl.store(out_ptrs, acc)

import torch
import triton
import triton.language as tl

def context_attention_fwd(Q, K, V, Out, BLOCK_M=128, BLOCK_DMODEL=64, BLOCK_N=64):
    # Ensure the head dimension Lk is a multiple of BLOCK_DMODEL
    Lk = Q.shape[1]
    assert Lk % BLOCK_DMODEL == 0, "Lk must be a multiple of BLOCK_DMODEL"

    # Compute the grid size
    grid = (triton.cdiv(Q.shape[0], BLOCK_M) * triton.cdiv(V.shape[1], BLOCK_N),)

    # Launch the kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        Lk, BLOCK_M, BLOCK_DMODEL, BLOCK_N
    )

# Example usage
Q = torch.randn(1024, 64, device='cuda')
K = torch.randn(1024, 64, device='cuda')
V = torch.randn(1024, 64, device='cuda')
Out = torch.empty_like(Q)

context_attention_fwd(Q, K, V, Out)

print(Out)
