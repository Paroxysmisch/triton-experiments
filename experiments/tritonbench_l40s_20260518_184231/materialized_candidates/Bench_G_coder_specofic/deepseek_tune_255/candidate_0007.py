import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator

@triton.jit
def ff_llama(
    x,
    w1,
    w3,
    w1_rms,
    w3_rms,
    y,
    lora_betas,
    lora_alphas,
    lora_groups,
    lora_group_ids,
    x_is_fp8,
    w1_is_fp8,
    w3_is_fp8,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_L2NORM_NORMALIZATION: tl.constexpr,
    SCALED_SIGMOID: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Fused forward kernel for llama.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x + rm[:, None] * N + rk[None, :] * 1
    w1_ptrs = w1 + rk[:, None] * K + rn[None, :] * 1
    w3_ptrs = w3 + rk[:, None] * K + rn[None, :] * 1

    lora_betas_ptrs = lora_betas + rn[None, :] * 1
    lora_alphas_ptrs = lora_alphas + rn[None, :] * 1
    lora_groups_ptrs = lora_groups + rn[None, :] * 1
    lora_group_ids_ptrs = lora_group_ids + rm[:, None] * 1

    w1_rms_ptrs = w1_rms + rn[None, :] * 1
    w3_rms_ptrs = w3_rms + rn[None, :] * 1

    x = tl.load(x_ptrs, mask=rk[None, :] < K, other=0.0)

    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        k_offset = k * BLOCK_SIZE_K
        w1_k_mask = (rk + k_offset)[None, :] < K
        w3_k_mask = w1_k_mask

        w1 = tl.load(w1_ptrs, mask=w1_k_mask, other=0.0)
        w3 = tl.load(w3_ptrs, mask=w3_k_mask, other=0.0)

        if k == 0:
            lora_betas = tl.load(lora_betas_ptrs, mask=rn[None, :] < N, other=0.0)
            lora_alphas = tl.load(lora_alphas_ptrs, mask=rn[None, :] < N, other=0.0)
            lora_groups = tl.load(lora_groups_ptrs, mask=rn[None, :] < N, other=0.0)
            lora_group_ids = tl.load(lora_group_ids_ptrs, mask=rm[:, None] < M, other=0.0)

        if x_is_fp8:
            x_fp8 = x
            w1_fp8 = w1
            w3_fp8 = w3
            if w1_is_fp8:
                w1_fp8 = tl.make_block_ptr(w1_fp8, (K,), (1,), (k_offset,), (1,), (0,))
            if w3_is_fp8:
                w3_fp8 = tl.make_block_ptr(w3_fp8, (K,), (1,), (k_offset,), (1,), (0,))

            acc1 += tl.dot(x_fp8, w1_fp8, allow_tf32=False, out_dtype=tl.float32)
            acc2 += tl.dot(x_fp8, w3_fp8, allow_tf32=False, out_dtype=tl.float32)
        else:
            acc1 += tl.dot(x, w1, allow_tf32=False, out_dtype=tl.float32)
            acc2 += tl.dot(x, w3, allow_tf32=False, out_dtype=tl.float32)

        w1_ptrs += BLOCK_SIZE_K
        w3_ptrs += BLOCK_SIZE_K

    acc1 = tl.where(rn[None, :] < N, acc1 * lora_betas, 0.0) + acc1 * lora_alphas

    if ACTIVATION == "silu":
        acc1 = tl.fdiv(acc1, 1 + tl.exp(-acc1))
    elif ACTIVATION == "relu":
        acc1 = tl.maximum(acc1, 0)

    w1_rms = tl.load(w1_rms_ptrs, mask=rn[None, :] < N, other=0.0)
    acc1 = acc1 / (w1_rms + EPS)

    acc2 = tl.where(rn[None, :] < N, acc2 * lora_betas, 0.0) + acc2 * lora_alphas

    if ACTIVATION == "silu":
        acc2 = tl.fdiv(acc2, 1 + tl.exp(-acc2))
    elif ACTIVATION == "relu":
        acc2 = tl.maximum(acc2, 0)

    w3_rms = tl.load(w3_rms_ptrs, mask=rn[None, :] < N, other=0.0)
    acc2 = acc2 / (w3_rms + EPS)

    if SCALED_SIGMOID:
        y = acc1 * tl.sigmoid(acc1)
    else:
        y = acc1

    if USE_L2NORM_NORMALIZATION:
        y_l2norm = tl.sqrt(tl.sum(y * y))
        y_l2norm = tl.maximum(y_l2norm, EPS)
        y = y / y_l2norm

    y = y * acc2

    y_ptrs = w3 + rm[:,
