import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'RBLOCK': 1}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 256, 'RBLOCK': 2}, num_stages=2, num_warps=4),
    ],
    key=['xnumel', 'rnumel']
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_ptr0,  # [xnumel, rnumel], float32
    in_ptr1,  # [rnumel], float32 (scale)
    in_ptr2,  # [rnumel], float32 (shift)
    in_numel,  # total number of elements (xnumel*rnumel)
    xnumel,    # number of rows
    rnumel,    # number of cols
    out_mean_ptr,    # [xnumel], float32
    out_invstd_ptr,  # [xnumel], float32
    out_ptr0,        # [xnumel, rnumel], float32
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr
):
    pid_x = tl.program_id(0)
    x_start = pid_x * XBLOCK
    x_offsets = x_start + tl.arange(0, XBLOCK)
    mask_x = x_offsets < xnumel

    mean_acc = tl.zeros([XBLOCK], dtype=tl.float32)
    var_acc = tl.zeros([XBLOCK], dtype=tl.float32)
    r_iter = tl.arange(0, RBLOCK)

    # Pass 1: compute mean and var across r dimension
    for r_base in range(0, rnumel, RBLOCK):
        r_offsets = r_base + r_iter
        mask_r = r_offsets < rnumel
        mask = mask_x[:, None] & mask_r[None, :]
        idx = x_offsets[:, None] * rnumel + r_offsets[None, :]
        data = tl.load(in_ptr0 + idx, mask=mask, other=0.0)
        sum_local = tl.sum(data, 1)
        sum_sq_local = tl.sum(data * data, 1)
        mean_acc += sum_local
        var_acc += sum_sq_local

    mean_acc = mean_acc / tl.float32(rnumel)
    var_acc = var_acc / tl.float32(rnumel) - mean_acc * mean_acc
    inv_std = 1.0 / tl.sqrt(var_acc + 1e-5)

    tl.store(out_mean_ptr + x_offsets, mean_acc, mask=mask_x)
    tl.store(out_invstd_ptr + x_offsets, inv_std, mask=mask_x)

    # Pass 2: apply normalization, scale, and shift
    for r_base in range(0, rnumel, RBLOCK):
        r_offsets = r_base + r_iter
        mask_r = r_offsets < rnumel
        mask = mask_x[:, None] & mask_r[None, :]
        idx = x_offsets[:, None] * rnumel + r_offsets[None, :]
        data = tl.load(in_ptr0 + idx, mask=mask, other=0.0)
        mean_val = tl.broadcast_to(mean_acc, [XBLOCK])
        inv_std_val = tl.broadcast_to(inv_std, [XBLOCK])
        norm_data = (data - mean_val[:, None]) * inv_std_val[:, None]
        scale = tl.load(in_ptr1 + r_offsets, mask=mask_r, other=1.0)
        shift = tl.load(in_ptr2 + r_offsets, mask=mask_r, other=0.0)
        out_data = norm_data * scale[None, :] + shift[None, :]
        tl.store(out_ptr0 + idx, out_data, mask=mask)

@torch.no_grad()
def fused_native_layer_norm_no_welford(
    input0: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
    out_mean: torch.Tensor,
    out_inv_std: torch.Tensor,
    out0: torch.Tensor,
    stream=None
):
    xnumel, rnumel = input0.shape
    in_numel = xnumel * rnumel
    assert input1.shape[0] == rnumel
    assert input2.shape[0] == rnumel
    assert out_mean.shape[0] == xnumel
    assert out_inv_std.shape[0] == xnumel
    grid = ( (xnumel + 127) // 128, )
    triton_red_fused_native_layer_norm_no_welford[grid](
        input0.data_ptr(),
        input1.data_ptr(),
        input2.data_ptr(),
        in_numel,
        xnumel,
        rnumel,
        out_mean.data_ptr(),
        out_inv_std.data_ptr(),
        out0.data_ptr(),
        stream=stream
    )
    return out0, out_mean, out_inv_std
