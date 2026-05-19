import torch
import triton
import triton.language as tl
from .tr_ops.tr_silu import silu
from .tr_ops.tr_layer_norm import trlayer_norm
from ..utils.shape_utils import volume


# fused 2d conv + layernorm + silu
@triton.jit
def fused_silu_ln_conv2d_fwd_fused(
    x_ptr,
    y_ptr,
    weight_ptr,
    ln_weight_ptr,
    conv_bias_ptr,
    x_row_stride,
    x_col_stride,
    x_batch_stride,
    y_row_stride,
    y_col_stride,
    y_batch_stride,
    M,
    N,
    K,
    BP,
    FP,
    S,
   _elems_per_thread: tl.constexpr,
    K_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    NORM_FIRST: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    pid_k = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUPS * num_pid_n
    group_id = pid_k // num_pid_in_group
    first_pid_n = group_id * num_pid_n
    group_size_n = min(num_pid_k * BLOCK_K, K - first_pid_k * BLOCK_K)
    pid_n = first_pid_n + ((pid_k % num_pid_in_group) % num_pid_n)
    pid_k = first_pid_k + ((pid_k % num_pid_in_group) // num_pid_n)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ck = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    y_offset = pid_b * y_batch_stride + pid_c * y_row_stride + cn * y_col_stride
    x_offset = pid_b * x_batch_stride + pid_c * x_row_stride + cn[:, None] * x_col_stride + ck[None, :] * FP
    y_ptrs = y_ptr + y_offset + tl.arange(0, elems_per_thread)
    x_ptrs = x_ptr + x_offset + tl.arange(0, elems_per_thread)
    ind = tl.arange(0, elems_per_thread)
    w = tl.load(weight_ptr + tl.arange(0, K_PER_GROUP))
    bias = tl.load(conv_bias_ptr + tl.arange(0, K_PER_GROUP))
    k_remaining = K - pid_k * BLOCK_K
    pad_w = tl.full((BLOCK_K,), 0.0, dtype=tl.float32)
    bpad_w = tl.full((BLOCK_K,), 0.0, dtype=tl.float32)
    if K_PER_GROUP < BLOCK_K:
        if pid_k < K / BLOCK_K:
            pad_w = tl.where(ck >= K, 0.0, tl.load(weight_ptr + ck, mask=ck < K))
            bpad_w = tl.where(ck >= K, 0.0, tl.load(conv_bias_ptr + ck, mask=ck < K))
        else:
            pass
        w = tl.where(ind[None, :] < k_remaining, w[None, :], pad_w[None, :])
        bias = tl.where(ind[None, :] < k_remaining, bias[None, :], bpad_w[None, :])
    else:
        w = tl.where(ind[None, :] < k_remaining, w[None, :], pad_w[None, :])
        bias = tl.where(ind[None, :] < k_remaining, bias[None, :], bpad_w[None, :])

    if GROUPS > 1:
        if K_PER_GROUP == 1:
            if pid_k < K / BLOCK_K:
                w = tl.broadcast_to(w, [BLOCK_K, GROUPS]).to(w.dtype)
                bias = tl.broadcast_to(bias, [BLOCK_K, GROUPS]).to(bias.dtype)
            else:
                w = tl.broadcast_to(pad_w, [BLOCK_K, GROUPS]).to(w.dtype)
                bias = tl.broadcast_to(bpad_w, [BLOCK_K, GROUPS]).to(bias.dtype)
        elif K_PER_GROUP != 1:
            w = tl.reshape(w, [K_PER_GROUP, GROUPS, BLOCK_K])
            w = tl.transpose(w, [1, 0, 2])
            bias = tl.reshape(bias, [K_PER_GROUP, GROUPS, BLOCK_K])
            bias = tl.transpose(bias, [1, 0, 2])
    if NORM_FIRST:
        I = tl.load(ln_weight_ptr + tl.arange(0, BLOCK_K)).to(tl.float32)
        x = tl.load(x_ptrs, mask=x_offset < (BP * S), other=0.0)
        x = x.to(tl.float32)
        mean = tl.sum(x, axis=1) / K
        x_zm = tl.where(ind < group_size_n, x - mean[:, None], 0.0)
        tl.store(y_ptrs, x_zm, mask=y_offset < (BP * S))
        y = tl.load(y_ptrs, mask=y_offset < (BP * S), other=0.0)
        y = y.to(tl.float32)
        var = tl.sum(x_zm * x_zm, axis=1) / K
        rstd = 1 / tl.sqrt(var + 1e-5)
        y_hat = y * rstd[:, None]
        y_final = y_hat * I[:, None]
        y_final += bias[None, :]
        y_final = silu(y_final)
        tl.store(y_ptr + y_offset, y_final, mask=y_offset < (BP * S))
    else:
        I = tl.load(ln_weight_ptr + tl.arange(0, BLOCK_K)).to(tl.float32)
        x = tl.load(x_ptrs, mask=x_offset < (BP * S), other=0.0)
        x = x.to(tl.float32)
        mean = tl.sum(x, axis=1) / K
        x_zm = tl.where(ind < group_size_n, x - mean[:, None], 0.0)
        y = x_zm.to(tl.float32)
        var = tl.sum(x_zm * x_zm, axis=1) / K
        rstd = 1 / tl.sqrt(var + 1e-5)
        y_hat = y * rstd[:, None]
        y_final = y_hat * I[:, None]
        y_final = silu(y_final)
        y_final += bias[None, :]
        tl.store(y_ptr + y_offset, y_final, mask=y_offset < (BP * S))


def fused_silu_layer_norm_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor = None,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    conv_groups: int = 1,
    ln_eps: float = 1e-5,
) -> torch.Tensor:
    # Check constraints.
    assert x.shape[-2:] == conv_weight.shape[-2:]
    assert x.is_contiguous()

    # allocate output
    y = torch.empty_like(x)

    # reshape input data into 2D tensor
    x_arg = x.view((-1, x.shape[-2], x.shape[-1]))
    M, N, K = x_arg.shape
    _, NP, FP = conv_weight.shape
    S = conv_dilation * (conv_weight.shape[-1] - 1) + 1
    conv_bias = (
        conv_bias.unsqueeze(0).expand((FP, 1)).clone().contiguous() if conv_bias is not None else None
    )  # shape now [out_channels, 1]

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    NUM_REGS = 64
    K_PER_GROUP = triton.cdiv(FP, conv_groups)
    BLOCK_K = min(triton.next_power_of_2(K_PER_GROUP), MAX_FUSED_SIZE)
    if K_PER_GROUP > NUM_REGS:
        BLOCK_K = triton.cdiv(NUM_REGS, K_PER_GROUP) * BLOCK_K
    BLOCK_N = triton.min_pow_of_2(MAX_FUSED_SIZE // BLOCK_K)
    if K <= MAX_FUSED_SIZE:
        grid = (M, N, triton.cdiv(K, BLOCK_K))
        num_stages = 4 if K <= 32 * 1024 else 3
        num_warps = 4
        fused_silu_ln_conv2d_fwd_fused[(M, N, triton.cdiv(K, BLOCK_K))](
            x_arg,
            y,
            conv_weight,
            weight,
            conv_bias,
            x_arg.stride(0),
            x_arg.stride(1),
            x_arg.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            M,
            N,
            K,
            FP,
            K_PER_GROUP,
            S=S,
            elems_per_thread=BLOCK_K,
            K_PER_GROUP=K_PER_GROUP,
            GROUPS=conv_groups,
            NORM_FIRST=True,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y.view
