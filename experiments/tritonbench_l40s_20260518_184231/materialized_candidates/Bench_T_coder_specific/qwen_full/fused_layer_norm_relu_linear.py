import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_relu_linear_fwd(
    input_ptr, weight_ptr, bias_ptr, ln_weight_ptr, ln_bias_ptr, pre_act_ptr, out_ptr,
    N,  # in_features
    C,  # out_features
    input_stride,  # input_ptr.stride(0)
    pre_act_stride,  # pre_act_ptr.stride(0)
    out_stride,  # out_ptr.stride(0)
    ln_weight_stride,  # ln_weight_ptr.stride(0)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    N_MID: tl.constexpr,
):
    pid = tl.program_id(0)
    # C: out_features, N: in_features
    # N_MID: mid feature size, reduce over mid
    # 1. mid feature fusion, reduce over N
    # 2. layer norm over mid feature
    # 3. linear layer over mid feature
    # 4. relu over linear layer
    # 5. final linear layer over relu
    # pid: 0 ~ M-1
    # block_id_n: 0 ~ (N // BLOCK_SIZE_N) - 1
    # block_id_c: 0 ~ (C // BLOCK_SIZE_C) - 1

    # block_id_n: 0 ~ (N // BLOCK_SIZE_N) - 1
    block_id_n = tl.program_id(0)
    # block_id_c: 0 ~ (C // BLOCK_SIZE_C) - 1
    block_id_c = tl.program_id(1)

    # mid feature fusion
    # accumulate at mid_feature_ptr
    # [C // BLOCK_SIZE_C, BLOCK_SIZE_N, BLOCK_SIZE_C] -> [BLOCK_SIZE_N, BLOCK_SIZE_C, C // BLOCK_SIZE_C]
    mid_feature = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_C, C // BLOCK_SIZE_C], dtype=tl.float32)

    # [BLOCK_SIZE_N, BLOCK_SIZE_C]
    input_ptr_mask = (block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None] * input_stride + \
        (block_id_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C))[None, :]  # [BLOCK_SIZE_N, BLOCK_SIZE_C]
    input = tl.load(input_ptr + input_ptr_mask,
                    mask=(block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None] <
                    input_ptr.shape[0] and (block_id_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C))[None, :] < input_ptr.shape[1],
                    other=0.0)

    # [C // BLOCK_SIZE_C, BLOCK_SIZE_N, BLOCK_SIZE_C] -> [BLOCK_SIZE_N, BLOCK_SIZE_C, C // BLOCK_SIZE_C]
    mid_feature += tl.transpose(input, [1, 0, 2])

    mid_feature = mid_feature.to(tl.float16)

    # linear
    # [BLOCK_SIZE_N, C]
    mid_feature = tl.reshape(mid_feature, [BLOCK_SIZE_N, C])
    linear = tl.dot(mid_feature, weight_ptr)

    if bias_ptr is not None:
        linear += bias_ptr

    # relu
    linear = tl.where(linear > 0, linear, 0)

    # save pre act
    pre_act_idx = pid * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    pre_act_ptr_mask = pre_act_idx[:, None] * pre_act_stride + \
        (block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]  # [BLOCK_SIZE_C, BLOCK_SIZE_N]
    tl.store(pre_act_ptr + pre_act_ptr_mask,
             linear, mask=pre_act_idx[:, None] < pre_act_ptr.shape[0] and (block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < pre_act_ptr.shape[1])

    # layer norm
    # [BLOCK_SIZE_N, BLOCK_SIZE_C]
    linear = tl.reshape(linear, [BLOCK_SIZE_N, BLOCK_SIZE_C])
    mean = tl.sum(linear, axis=0) / N

    # [BLOCK_SIZE_N, BLOCK_SIZE_C]
    linear = tl.where((block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None] <
                      linear.shape[0] and (block_id_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C))[None, :] < linear.shape[1], linear, 0.0)

    # [BLOCK_SIZE_C]
    # keepdim?
    mean = tl.sum(linear, axis=0) / N
    # [BLOCK_SIZE_N, BLOCK_SIZE_C]
    linear = linear - mean
    # [BLOCK_SIZE_C]
    var = tl.sum(linear * linear, axis=0) / N
    rstd = 1 / tl.sqrt(var + 1e-5)

    linear = linear * rstd

    # affine
    if ln_weight_ptr is not None and ln_bias_ptr is not None:
        linear = linear * ln_weight_ptr + ln_bias_ptr

    # [BLOCK_SIZE_N, BLOCK_SIZE_C]
    linear = tl.reshape(linear, [BLOCK_SIZE_N, BLOCK_SIZE_C])
    out_idx = pid * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    out_ptr_mask = out_idx[:, None] * out_stride + \
        (block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]  # [BLOCK_SIZE_C, BLOCK_SIZE_N]
    tl.store(out_ptr + out_ptr_mask,
             linear, mask=out_idx[:, None] < out_ptr.shape[0] and (block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < out_ptr.shape[1])


def fused_layer_norm_relu_linear_triton(input, weight, bias=None, ln_weight=None, ln_bias=None, eps=1e-5, elementwise_affine=True):
    # input: [M, N], weight: [C, N], bias: [C], ln_weight: [C], ln_bias: [C]
    # output: [M, C]
    assert input.shape[1] == weight.shape[1], "in_features must match"
    assert input.is_contiguous(), "input must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert bias is None or bias.is_contiguous(), "bias must be None or contiguous"
    assert ln_weight is None or ln_weight.is_contiguous(), "ln_weight must be None or contiguous"
    assert ln_bias is None or ln_bias.is_contiguous(), "ln_bias must be contiguous"

    M, N = input.shape
    C = weight.shape[0]

    BLOCK_SIZE_N = 8
    BLOCK_SIZE_C = 16
    # 16, 32, 64, 128, 256
    # 2, 4, 8, 16, 32
    # 1/8/16 input
    # 1/4/8/16 mid_feature
    # 1/2/4/8/16/32/64/128/256 output
    # BLOCK_SIZE_N * BLOCK_SIZE_C >= C
    # N_MID * BLOCK_SIZE_N == N
    N_MID = N // BLOCK_SIZE_N

    # [M, C]
    out = torch.empty((M, C), device=input.device, dtype=input.dtype)
    # [M, N]
    pre_act = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = (triton.cdiv(N, BLOCK_SIZE_N) * triton.cdiv(C, BLOCK_SIZE_C), )

    # [M, N], [C, N], [C], [C], [C], [M, N], [M, C]
    _fused_layer_norm_relu_linear_fwd[grid](
        input, weight, bias, ln_weight, ln_bias, pre_act, out,
        N, C,
        input.stride(0),
        pre_act.stride(0),
        out.stride(0),
        ln_weight.stride(0) if ln_weight is not None else 0,
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
        N_MID,
        num_warps=4,
        num_stages=2,
    )

    return out
