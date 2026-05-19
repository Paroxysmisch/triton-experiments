import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(
    X_ptr, Y_ptr, Z_ptr, M_ptr, O_ptr, weight_ptr, bias_ptr,
    stride_xn, stride_xc, stride_yn, stride_yc, stride_zn, stride_zc, stride_mn, stride_mc, stride_on, stride_oc,
    stride_w, stride_b,
    N, C, num_groups, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N * C
    x_offsets = tl.arange(0, N)[:, None] * stride_xn + tl.arange(0, C)[None, :] * stride_xc
    y_offsets = tl.arange(0, N)[:, None] * stride_yn + tl.arange(0, C)[None, :] * stride_yc
    z_offsets = tl.arange(0, N)[:, None] * stride_zn + tl.arange(0, C)[None, :] * stride_zc
    m_offsets = tl.arange(0, N)[:, None] * stride_mn + tl.arange(0, C)[None, :] * stride_mc
    o_offsets = tl.arange(0, N)[:, None] * stride_on + tl.arange(0, C)[None, :] * stride_oc

    x = tl.load(X_ptr + x_offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + y_offsets, mask=mask, other=0.0)

    z = x + y
    m = z * y

    group_size = C // num_groups
    group_id = (offsets % (N * C)) // (group_size * N)
    group_offset = group_id * group_size

    mean = tl.zeros((N, num_groups), dtype=tl.float32)
    var = tl.zeros((N, num_groups), dtype=tl.float32)

    for i in range(group_size):
        group_idx = group_offset + i
        group_mask = (offsets % (N * C)) == group_idx
        group_m = m[group_mask]
        mean += tl.sum(group_m, axis=1)
        var += tl.sum((group_m - mean) ** 2, axis=1)

    mean /= group_size
    var /= group_size

    std = tl.sqrt(var + eps)
    o = (m - mean) / std

    weight = tl.load(weight_ptr + offsets % C, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets % C, mask=mask, other=0.0)

    o = o * weight + bias

    tl.store(O_ptr + o_offsets, o, mask=mask)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N', 'C']
)
@triton.jit
def fused_add_mul_groupnorm_kernel(
    X_ptr, Y_ptr, Z_ptr, M_ptr, O_ptr, weight_ptr, bias_ptr,
    stride_xn, stride_xc, stride_yn, stride_yc, stride_zn, stride_zc, stride_mn, stride_mc, stride_on, stride_oc,
    stride_w, stride_b,
    N, C, num_groups, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Kernel implementation as provided above
    pass

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    N, C = input1.shape
    assert input1.shape == input2.shape, "Input tensors must have the same shape"
    assert weight.shape == (C,), "Weight must have shape (C,)"
    assert bias.shape == (C,), "Bias must have shape (C,)"
    assert C % num_groups == 0, "Number of groups must divide the number of channels evenly"

    if out is None:
        out = torch.empty_like(input1)

    X_ptr = input1.data_ptr()
    Y_ptr = input2.data_ptr()
    Z_ptr = triton.empty_like(input1).data_ptr()
    M_ptr = triton.empty_like(input1).data_ptr()
    O_ptr = out.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr()

    stride_xn, stride_xc = input1.stride()
    stride_yn, stride_yc = input2.stride()
    stride_zn, stride_zc = triton.empty_like(input1).stride()
    stride_mn, stride_mc = triton.empty_like(input1).stride()
    stride_on, stride_oc = out.stride()
    stride_w = weight.stride(0)
    stride_b = bias.stride(0)

    grid = (N * C // 128, 1, 1)
    fused_add_mul_groupnorm_kernel[grid](
        X_ptr, Y_ptr, Z_ptr, M_ptr, O_ptr, weight_ptr, bias_ptr,
        stride_xn, stride_xc, stride_yn, stride_yc, stride_zn, stride_zc, stride_mn, stride_mc, stride_on, stride_oc,
        stride_w, stride_b,
        N, C, num_groups, eps,
        BLOCK_SIZE=128
    )

    return out
