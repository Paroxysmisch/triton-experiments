import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    N_CTX, N_HEAD, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_in_block = num_pid_m * num_pid_n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh)
    k_ptrs = K + (offs_n[:, None] * stride_km + offs_d[None, :] * stride_kh)
    v_ptrs = V + (offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vh)
    o_ptrs = Out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_oh)
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d in range(0, N_HEAD, BLOCK_D):
        qk = tl.dot(q, k, trans_b=True)
        acc += qk * SCALE
    tl.store(o_ptrs, acc)

@triton.jit
def parallel_retention_bwd_kernel(
    Q, K, V, GradOut, GradQ, GradK, GradV,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_gob, stride_goh, stride_gom,
    stride_gqb, stride_gqh, stride_gqm,
    stride_gkb, stride_gkh, stride_gkm,
    stride_gvb, stride_gvh, stride_gvm,
    N_CTX, N_HEAD, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_in_block = num_pid_m * num_pid_n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh)
    k_ptrs = K + (offs_n[:, None] * stride_km + offs_d[None, :] * stride_kh)
    v_ptrs = V + (offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vh)
    go_ptrs = GradOut + (offs_m[:, None] * stride_gom + offs_n[None, :] * stride_goh)
    gq_ptrs = GradQ + (offs_m[:, None] * stride_gqm + offs_d[None, :] * stride_gqh)
    gk_ptrs = GradK + (offs_n[:, None] * stride_gkm + offs_d[None, :] * stride_gkh)
    gv_ptrs = GradV + (offs_n[:, None] * stride_gvm + offs_d[None, :] * stride_gvh)
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    go = tl.load(go_ptrs)
    gq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    gk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    gv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for d in range(0, N_HEAD, BLOCK_D):
        qk = tl.dot(q, k, trans_b=True)
        qk *= SCALE
        qk *= go
        gq += tl.dot(qk, v, trans_b=True)
        gk += tl.dot(q, qk, trans_a=True)
        gv += tl.dot(qk, k)
    tl.store(gq_ptrs, gq)
    tl.store(gk_ptrs, gk)
    tl.store(gv_ptrs, gv)

import torch

def forward(q, k, v, scale, block_m=16, block_n=16, block_d=16):
    N_CTX, N_HEAD, _ = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, block_m) * triton.cdiv(N_CTX, block_n),)
    parallel_retention_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N_CTX, N_HEAD, scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d
    )
    return out

def backward(q, k, v, grad_out, scale, block_m=16, block_n=16, block_d=16):
    N_CTX, N_HEAD, _ = q.shape
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    grid = (triton.cdiv(N_CTX, block_m) * triton.cdiv(N_CTX, block_n),)
    parallel_retention_bwd_kernel[grid](
        q, k, v, grad_out, grad_q, grad_k, grad_v,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
        grad_k.stride(0), grad_k.stride(1), grad_k.stride(2),
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
        N_CTX, N_HEAD, scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d
    )
    return grad_q, grad_k, grad_v

import torch

# Example tensors
N_CTX = 128
N_HEAD = 8
D_HEAD = 64
q = torch.randn(N_CTX, N_HEAD, D_HEAD, device='cuda')
k = torch.randn(N_CTX, N_HEAD, D_HEAD, device='cuda')
v = torch.randn(N_CTX, N_HEAD, D_HEAD, device='cuda')
scale = 1.0 / (D_HEAD ** 0.5)

# Forward pass
out = forward(q, k, v, scale)

# Backward pass
grad_out = torch.randn_like(out)
grad_q, grad_k, grad_v = backward(q, k, v, grad_out, scale)

print(out)
print(grad_q)
print(grad_k)
print(grad_v)
