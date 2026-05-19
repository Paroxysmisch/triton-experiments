import triton
import triton.language as tl
import numpy as np
import torch

@triton.jit
def _fwd_kernel_int8kv(
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    Z: tl.constexpr, H: tl.constexpr,
    Q, K, V, Scales, Output,
    offs_m: tl.constexpr, offs_n: tl.constexpr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    start_m, causal_mask, head_dim
):
    qvk_offset = start_m * (BLOCK_M * BLOCK_N)
    Q_ptrs = Q + qvk_offset
    K_ptrs = K + qvk_offset
    V_ptrs = V + qvk_offset
    S_ptrs = Scales + qvk_offset
    Off_m_i = offs_m + head_dim * start_m
    Off_n_i = offs_n

    # Load Q, K, and V values
    q = tl.load(Q_ptrs)
    k = tl.load(K_ptrs)
    v = tl.load(V_ptrs)

    # Compute the dot product of Q and K
    dot_product = tl.dot(q, k)

    # Scale the dot product using the softmax scaling factor
    qk_scaled = dot_product * S_ptrs

    # Apply the causal mask
    out = qk_scaled * causal_mask

    # Apply the softmax function to get the attention probabilities
    attn_probs = tl.softmax(out)

    # Use the attention probabilities to weigh the values in V
    weighted_vals = attn_probs * v

    # Sum up the weighted values along dimension n
    output = tl.sum(weighted_vals, (BLOCK_N,))

    # Store the result in the output tensor
    tl.store(Output, output)


def context_attention_fwd_ppl_int8kv(Q, K, V, Scales, Output, causal_mask):
    BLOCK_M = 128
    BLOCK_N = 64

    # Based on the input tensor dimensions, set parameters
    Z, H, _ = Q.shape
    total_elements = Z * H

    # Configure the execution grid
    n_warps = 4
    n_stages = 4
    grid = (triton.cdiv(total_elements, n_warps * BLOCK_M), 1, 1)

    # Invoke the kernel
    _fwd_kernel_int8kv[grid](
        BLOCK_M, BLOCK_N,
        Z, H,
        Q, K, V, Scales, Output,
        stride_qz=Q.stride(0), stride_qh=Q.stride(1), stride_qm=Q.stride(2), stride_qk=Q.stride(3),
        stride_kz=K.stride(0), stride_kh=K.stride(1), stride_kn=K.stride(2), stride_kk=K.stride(3),
        stride_vz=V.stride(0), stride_vh=V.stride(1), stride_vk=V.stride(2), stride_vn=V.stride(3),
        stride_oz=Output.stride(0), stride_oh=Output.stride(1), stride_om=Output.stride(2), stride_on=Output.stride(3),
        start_m=triton.program_id(0), causal_mask=causal_mask, head_dim=V.shape[-1],
        num_warps=n_warps, num_stages=n_stages
    )
