import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, O, 
    stride_qm, stride_qh, stride_qd, 
    stride_km, stride_kh, stride_kd, 
    stride_vm, stride_vh, stride_vd, 
    stride_om, stride_oh, stride_od, 
    M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, 
    sm_scale: tl.constexpr, 
    IS_CAUSAL: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, 
    USE_FP8: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_warp = BLOCK_DMODEL // 16
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    head_id = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + head_id * stride_qh + offs_d[None, :] * stride_qd)
    k_ptrs = K + (offs_n[:, None] * stride_km + head_id * stride_kh + offs_d[None, :] * stride_kd)
    v_ptrs = V + (offs_n[:, None] * stride_vm + head_id * stride_vh + offs_d[None, :] * stride_vd)
    o_ptrs = O + (offs_m[:, None] * stride_om + head_id * stride_oh + offs_d[None, :] * stride_od)
    q = tl.load(q_ptrs, boundary_check=(0, 1))
    k = tl.load(k_ptrs, boundary_check=(0, 1))
    v = tl.load(v_ptrs, boundary_check=(0, 1))
    q = q * sm_scale

    # Compute dot product
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        k = tl.load(k_ptrs + n * BLOCK_N * stride_kd, boundary_check=(0, 1))
        v = tl.load(v_ptrs + n * BLOCK_N * stride_vd, boundary_check=(0, 1))
        if IS_CAUSAL:
            mask = offs_m[:, None] >= (offs_n + n * BLOCK_N)[None, :]
            qk = tl.where(mask, tl.dot(q, k, allow_tf32=False), -float('inf'))
        else:
            qk = tl.dot(q, k, allow_tf32=False)
        if USE_FP8:
            qk = qk.to(tl.float8)
        acc += tl.dot(qk, v, allow_tf32=False)

    # Store the result
    tl.store(o_ptrs, acc, boundary_check=(0, 1))

class TritonSelfAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale, IS_CAUSAL, USE_FP8):
        B, H, M, D = Q.shape
        N = K.shape[2]
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_DMODEL = 128
        num_warps = 4 if BLOCK_DMODEL <= 128 else 8
        num_stages = 2

        O = torch.empty((B, H, M, D), device=Q.device, dtype=Q.dtype)
        grid = (triton.cdiv(M, BLOCK_M), H * B)

        _fwd_kernel[grid](
            Q, K, V, O, 
            Q.stride(2), Q.stride(1), Q.stride(3), 
            K.stride(2), K.stride(1), K.stride(3), 
            V.stride(2), V.stride(1), V.stride(3), 
            O.stride(2), O.stride(1), O.stride(3), 
            M, N, H, D, 
            sm_scale, 
            IS_CAUSAL, 
            BLOCK_M, BLOCK_N, BLOCK_DMODEL, 
            USE_FP8, 
            num_warps=num_warps, 
            num_stages=num_stages
        )

        ctx.save_for_backward(Q, K, V, O)
        ctx.sm_scale = sm_scale
        ctx.IS_CAUSAL = IS_CAUSAL
        ctx.USE_FP8 = USE_FP8
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        IS_CAUSAL = ctx.IS_CAUSAL
        USE_FP8 = ctx.USE_FP8
        B, H, M, D = Q.shape
        N = K.shape[2]
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_DMODEL = 128
        num_warps = 4 if BLOCK_DMODEL <= 128 else 8
        num_stages = 2

        grad_Q = torch.empty_like(Q)
        grad_K = torch.empty_like(K)
        grad_V = torch.empty_like(V)

        # Backward pass kernel (not provided in the query, so this is a placeholder)
        # _bwd_kernel[grid](
        #     Q, K, V, O, grad_output, grad_Q, grad_K, grad_V, 
        #     Q.stride(2), Q.stride(1), Q.stride(3), 
        #     K.stride(2), K.stride(1), K.stride(3), 
        #     V.stride(2), V.stride(1), V.stride(3), 
        #     O.stride(2), O.stride(1), O.stride(3), 
        #     grad_output.stride(2), grad_output.stride(1), grad_output.stride(3), 
        #     grad_Q.stride(2), grad_Q.stride(1), grad_Q.stride(3), 
        #     grad_K.stride(2), grad_K.stride(1), grad_K.stride(3), 
        #     grad_V.stride(2), grad_V.stride(1), grad_V.stride(3), 
        #     M, N, H, D, 
        #     sm_scale, 
        #     IS_CAUSAL, 
        #     BLOCK_M, BLOCK_N, BLOCK_DMODEL, 
        #     USE_FP8, 
        #     num_warps=num_warps, 
        #     num_stages=num_stages
        # )

        return grad_Q, grad_K, grad_V, None, None, None

triton_self_attention = TritonSelfAttentionFunction.apply

import torch

# Example tensors
B, H, M, D = 2, 4, 128, 64
Q = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
K = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
V = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
sm_scale = 1.0 / (D ** 0.5)
IS_CAUSAL = True
USE_FP8 = False

# Forward pass
O = triton_self_attention(Q, K, V, sm_scale, IS_CAUSAL, USE_FP8)

print(O.shape)  # Should print: torch.Size([2, 4, 128, 64])
