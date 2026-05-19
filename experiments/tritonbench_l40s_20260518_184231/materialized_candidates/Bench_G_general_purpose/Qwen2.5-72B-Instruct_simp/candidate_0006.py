import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    stride_b0b, stride_b0h, stride_b0m,
    nheads, size_m, size_n, size_k,
    sm_scale, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(size_m, block_m)
    num_pid_n = tl.cdiv(size_n, block_n)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % num_pid_in_group) // num_pid_m

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_h = tl.arange(0, nheads)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_h[None, :] * stride_qh)
    k_ptrs = K + (offs_n[None, :] * stride_kn + offs_h[:, None] * stride_kh)
    v_ptrs = V + (offs_n[None, :] * stride_vm + offs_h[:, None] * stride_vh)
    b0_ptrs = B0 + (offs_m[:, None] * stride_b0m + offs_h[None, :] * stride_b0h)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for start_n in range(0, size_n, block_n):
        k_block_ptr = k_ptrs + start_n * stride_kn
        v_block_ptr = v_ptrs + start_n * stride_vm
        b0_block_ptr = b0_ptrs + start_n * stride_b0m

        q = tl.load(q_ptrs)
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        b0 = tl.load(b0_block_ptr)

        qk = tl.dot(q, k, trans_b=True)
        qk += b0
        qk *= sm_scale
        qk = tl.softmax(qk, axis=1)

        o = tl.dot(qk, v)
        acc += o

    out_ptrs = Out + (offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh)
    tl.store(out_ptrs, acc)

import torch

def _attention_rel_h_rel_w_kernel_aligned_device(Q, K, V, B0, Out, sm_scale):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Get the dimensions of the input tensors
    batch_size, nheads, size_m, size_k = Q.shape
    size_n = K.shape[2]

    # Define the grid and block dimensions
    grid = (triton.cdiv(size_m, BLOCK_M) * triton.cdiv(size_n, BLOCK_N) * nheads,)

    # Define the strides for the input and output tensors
    stride_qb = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_qm = Q.stride(2)
    stride_kb = K.stride(0)
    stride_kh = K.stride(1)
    stride_kn = K.stride(2)
    stride_vb = V.stride(0)
    stride_vh = V.stride(1)
    stride_vm = V.stride(2)
    stride_ob = Out.stride(0)
    stride_oh = Out.stride(1)
    stride_om = Out.stride(2)
    stride_b0b = B0.stride(0)
    stride_b0h = B0.stride(1)
    stride_b0m = B0.stride(2)

    # Launch the kernel
    _fwd_kernel_aligned[grid](
        Q, K, V, B0, Out,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vm,
        stride_ob, stride_oh, stride_om,
        stride_b0b, stride_b0h, stride_b0m,
        nheads, size_m, size_n, size_k,
        sm_scale, BLOCK_M, BLOCK_N, BLOCK_K
    )
