import triton
import triton.language as tl
import paddle
from paddle import Tensor
from typing import Optional, Tuple

@triton.autotune(
    configs=[
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "BACKWARD_PASS": False}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 64, "BACKWARD_PASS": False}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 16, "BACKWARD_PASS": False}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "BACKWARD_PASS": True}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 64, "BACKWARD_PASS": True}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 16, "BACKWARD_PASS": True}),
    ],
    key=["V", "N", "H", "COS", "SIN", "BACKWARD_PASS"],
)
@triton.jit
def _triton_rope(
    q_ptr,
    k_ptr,
    dq_ptr,
    dk_ptr,
    cos,
    sin,
    stride_q1,
    stride_q2,
    stride_q3,
    stride_q4,
    stride_k1,
    stride_k2,
    stride_k3,
    stride_k4,
    stride_dq1,
    stride_dq2,
    stride_dq3,
    stride_dq4,
    stride_dk1,
    stride_dk2,
    stride_dk3,
    stride_dk4,
    cos_stride_1,
    cos_stride_2,
    sin_stride_1,
    sin_stride_2,
    pid,
    H,
    V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr,
    COS_HAS_SPLIT: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    # Triton kernel to apply rotary position embeddings to query and key matrices
    v_pid = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)
    h_pid = tl.program_id(axis=2)
    v_offs = v_pid * V_BLOCK_SIZE
    n_offs = n_pid * N_BLOCK_SIZE
    h_offs = h_pid * H_BLOCK_SIZE

    cos_offs_1 = (v_offs // C_BLOCK_SIZE) * cos_stride_1
    cos_offs_2 = (h_offs // C_BLOCK_SIZE) * cos_stride_2
    sin_offs_1 = (v_offs // C_BLOCK_SIZE) * sin_stride_1
    sin_offs_2 = (h_offs // C_BLOCK_SIZE) * sin_stride_2

    if COS_HAS_SPLIT:
        cos_ptr = cos + pid
        cos_a = tl.load(cos_ptr + cos_offs_1)
        cos_b = tl.load(cos_ptr + cos_offs_2)
        sin_a = tl.load(cos_ptr + sin_offs_1)
        sin_b = tl.load(cos_ptr + sin_offs_2)
    else:
        cos_a = tl.load(cos + cos_offs_1)
        cos_b = tl.load(cos + cos_offs_2)
        sin_a = tl.load(cos + sin_offs_1)
        sin_b = tl.load(cos + sin_offs_2)

    offs_q1 = n_offs + tl.arange(0, N_BLOCK_SIZE)
    offs_q2 = h_offs + tl.arange(0, H_BLOCK_SIZE // 2)
    offs_q3 = v_offs + tl.arange(0, V_BLOCK_SIZE)
    offs_k1 = n_offs + tl.arange(0, N_BLOCK_SIZE)
    offs_k2 = h_offs + tl.arange(0, H_BLOCK_SIZE // 2)
    offs_k3 = v_offs + tl.arange(0, V_BLOCK_SIZE)

    offs_dq1 = n_offs + tl.arange(0, N_BLOCK_SIZE)
    offs_dq2 = h_offs + tl.arange(0, H_BLOCK_SIZE // 2)
    offs_dq3 = v_offs + tl.arange(0, V_BLOCK_SIZE)
    offs_dk1 = n_offs + tl.arange(0, N_BLOCK_SIZE)
    offs_dk2 = h_offs + tl.arange(0, H_BLOCK_SIZE // 2)
    offs_dk3 = v_offs + tl.arange(0, V_BLOCK_SIZE)

    offs_q4 = pid
    offs_k4 = pid
    offs_dq4 = pid
    offs_dk4 = pid

    mask_q2 = offs_q2 < H
    mask_q3 = offs_q3 < V_BLOCK_SIZE
    mask_q = mask_q2 & mask_q3

    mask_k2 = offs_k2 < H
    mask_k3 = offs_k3 < V_BLOCK_SIZE
    mask_k = mask_k2 & mask_k3

    mask_dq2 = offs_dq2 < H
    mask_dq3 = offs_dq3 < V_BLOCK_SIZE
    mask_dq = mask_dq2 & mask_dq3

    mask_dk2 = offs_dk2 < H
    mask_dk3 = offs_dk3 < V_BLOCK_SIZE
    mask_dk = mask_dk2 & mask_dk3

    q_1_ptr = q_ptr + stride_q1 * offs_q4 + stride_q2 * offs_q1[:, None] + stride_q3 * offs_q2[None, :]
    q_2_ptr = q_ptr + stride_q1 * offs_q4 + stride_q2 * offs_q1[:, None] + stride_q3 * offs_q3[None, :]

    k_1_ptr = k_ptr + stride_k1 * offs_k4 + stride_k2 * offs_k1[:, None] + stride_k3 * offs_k2[None, :]
    k_2_ptr = k_ptr + stride_k1 * offs_k4 + stride_k2 * offs_k1[:, None] + stride_k3 * offs_k3[None, :]

    dq_1_ptr = dq_ptr + stride_dq1 * offs_dq4 + stride_dq2 * offs_dq1[:, None] + stride_dq3 * offs_dq2[None, :]
    dq_2_ptr = dq_ptr + stride_dq1 * offs_dq4 + stride_dq2 * offs_dq1[:, None] + stride_d
