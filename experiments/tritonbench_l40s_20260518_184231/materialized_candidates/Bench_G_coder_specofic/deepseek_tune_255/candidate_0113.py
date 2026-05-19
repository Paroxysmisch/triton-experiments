import torch
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    primals_3, primals_1, primals_2, out_ptr0, out_ptr1, stride_0_0, stride_0_1,
    RBLOCK: tl.constexpr, S: tl.constexpr, D: tl.constexpr, eps: tl.constexpr, affine: tl.constexpr
):
    # Triton kernel for layer normalization
    row_block_id = tl.program_id(0)
    out_ptr0 += row_block_id * S
    out_ptr1 += row_block_id * S
    col_offsets = tl.arange(0, RBLOCK)
    mask = col_offsets < D
    row_block_ids = tl.full([RBLOCK], row_block_id, dtype=tl.int32)
    # Welford algorithm for mean and variance
    _mean = tl.zeros([RBLOCK], dtype=tl.float32)
    _m2 = tl.zeros([RBLOCK], dtype=tl.float32)
    _weight = tl.zeros([RBLOCK], dtype=tl.float32)
    for col_block_id in range(0, tl.cdiv(D, RBLOCK)):
        x = tl.load(primals_3 + row_block_ids * stride_0_0 + col_block_id * RBLOCK + col_offsets, mask=mask, other=0).to(tl.float32)
        _weight += tl.where(mask, 1, 0)
        x_mean = tl.where(mask, x, 0)
        x_mean = tl.sum(x_mean) / D
        x_var = tl.where(mask, (x - x_mean) * (x - x_mean), 0)
        x_var = tl.sum(x_var) / D
        _mean += x_mean
        _m2 += x_var
    _mean = tl.sum(_mean) / D
    _m2 = tl.sum(_m2) / D
    tmp3_mean = _mean
    tmp3_m2 = _m2
    tmp3_weight = _weight
    tmp5 = tmp3_weight
    tmp4 = tl.sqrt(tmp3_m2 + eps)
    tl.store(out_ptr0 + col_offsets, tmp3_mean, mask=mask)
    tl.store(out_ptr0 + D + col_offsets, tmp4, mask=mask)
    tl.store(out_ptr0 + 2 * D + col_offsets, tmp5, mask=mask)
    # Normalization
    if affine:
        _scale = tl.load(primals_1 + col_offsets, mask=mask, other=0)
        _bias = tl.load(primals_2 + col_offsets, mask=mask, other=0)
    for col_block_id in range(0, tl.cdiv(D, RBLOCK)):
        x = tl.load(primals_3 + row_block_ids * stride_0_0 + col_block_id * RBLOCK + col_offsets, mask=mask, other=0).to(tl.float32)
        tmp10 = (x - tmp3_mean) / tmp4
        if affine:
            tmp11 = tmp10 * _scale
            tmp11 = tmp11 + _bias
        else:
            tmp11 = tmp10
        tl.store(out_ptr1 + col_block_id * RBLOCK + col_offsets, tmp11, mask=mask)

def fused_native_layer_norm(x, weight, bias, eps, affine):
    # Python wrapper for invoking the Triton kernel
    B, S, D = x.shape
    buf0 = torch.empty((B, S, D), dtype=torch.float32, device=x.device)
    buf3 = torch.empty((B, S, D), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(S, meta['RBLOCK']),)
    rfactor = triton.ReplicationFactor(B, 0)
    with torch.cuda.device(x.device.index):
        triton_red_fused_native_layer_norm_0[grid](x, weight, bias, buf0, buf3,
                                                   x.stride(0), x.stride(1),
                                                   S, D, eps, affine,
                                                   replication_factor=rfactor)
    return buf0, buf3
