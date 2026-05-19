import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr,
    dq_ptr, dk_ptr, dg_ptr,
    M, N, H, W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_pid_w = tl.cdiv(W, BLOCK_W)
    num_pid_in_w = num_pid_m * num_pid_n * num_pid_h
    pid_in_w = pid % num_pid_in_w
    pid_w = pid // num_pid_in_w
    pid_m = pid_in_w // (num_pid_n * num_pid_h)
    pid_n = (pid_in_w % (num_pid_n * num_pid_h)) // num_pid_h
    pid_h = pid_in_w % num_pid_h

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    q = tl.load(q_ptr + offs_m[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :])
    k = tl.load(k_ptr + offs_n[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :])
    v = tl.load(v_ptr + offs_n[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :])
    g = tl.load(g_ptr + offs_m[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :])
    h = tl.load(h_ptr + offs_n[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :])
    do = tl.load(do_ptr + offs_m[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :])

    # Compute intermediate results
    b_dq = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_W), dtype=tl.float32)
    b_dk = tl.zeros((BLOCK_N, BLOCK_H, BLOCK_W), dtype=tl.float32)
    b_dg = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_W), dtype=tl.float32)

    for i in range(BLOCK_N):
        b_dq += tl.sum(do * v[i, :, :, :], axis=0)
        b_dk += tl.sum(q * do[i, :, :, :], axis=0)
        b_dg += tl.sum(do * h[i, :, :, :], axis=0)

    # Store results
    tl.store(dq_ptr + offs_m[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :], b_dq)
    tl.store(dk_ptr + offs_n[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :], b_dk)
    tl.store(dg_ptr + offs_m[:, None, None, None] * H * W + offs_h[None, :, None, None] * W + offs_w[None, None, :], b_dg)

import torch
import triton
import triton.language as tl

def chunk_bwd_dqkg_fn(q, k, v, g, h, do, dq, dk, dg, M, N, H, W, BLOCK_M, BLOCK_N, BLOCK_H, BLOCK_W):
    # Define grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']) * triton.cdiv(H, meta['BLOCK_H']) * triton.cdiv(W, meta['BLOCK_W']),
    )

    # Launch the Triton kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, g, h, do,
        dq, dk, dg,
        M, N, H, W,
        BLOCK_M, BLOCK_N, BLOCK_H, BLOCK_W
    )

# Example usage
M, N, H, W = 128, 128, 64, 64
BLOCK_M, BLOCK_N, BLOCK_H, BLOCK_W = 16, 16, 16, 16

q = torch.randn((M, H, W), device='cuda', dtype=torch.float32)
k = torch.randn((N, H, W), device='cuda', dtype=torch.float32)
v = torch.randn((N, H, W), device='cuda', dtype=torch.float32)
g = torch.randn((M, H, W), device='cuda', dtype=torch.float32)
h = torch.randn((N, H, W), device='cuda', dtype=torch.float32)
do = torch.randn((M, H, W), device='cuda', dtype=torch.float32)

dq = torch.zeros((M, H, W), device='cuda', dtype=torch.float32)
dk = torch.zeros((N, H, W), device='cuda', dtype=torch.float32)
dg = torch.zeros((M, H, W), device='cuda', dtype=torch.float32)

chunk_bwd_dqkg_fn(q, k, v, g, h, do, dq, dk, dg, M, N, H, W, BLOCK_M, BLOCK_N, BLOCK_H, BLOCK_W)
