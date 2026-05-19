import torch
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
        primals_3, primals_0, primals_1, primals_2, out,
        S, D, RBLOCK: tl.constexpr):
    # The Triton kernel for layer normalization.
    #
    # The layer norm is applied along the second dimension, `D`.
    # Each block processes a different segment of the second dimension, panning
    # across multiple features.
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK) % D
    block_idx = tl.program_id(0) * RBLOCK + tl.arange(0, RBLOCK)

    # Load the intermediate results buffers
    buf0 = tl.load(primals_0 + row, mask=block_idx < S)
    buf3 = tl.load(primals_3 + row, mask=block_idx < S).to(tl.float32)
    # compute mean and variance
    count = tl.zeros([RBLOCK], tl.int64)
    partial_mean = tl.zeros([RBLOCK], tl.float32)
    partial_m2 = tl.zeros([RBLOCK], tl.float32)
    mask = cols < D
    for inc in range(0, tl.cdiv(D, RBLOCK)):
        x = tl.load(buf3 + inc * RBLOCK + cols, mask=mask).to(tl.float32)
        count += tl.minimum(D - inc * RBLOCK, mask)
        old_mean = partial_mean
        partial_mean += (x - partial_mean) / count
        partial_m2 += (x - old_mean) * (x - partial_mean)
    tmp3_mean = partial_mean / D
    tmp5 = count.to(tl.float32)
    tmp3_m2 = partial_m2 / D

    # create offsets for stores
    buf4 = out + row * D + block_idx
    tl.store(buf0 + row, tmp5, mask=block_idx < S)
    tl.store(buf3 + row, tmp3_mean, mask=block_idx < S)
    tmp4 = tl.where(tmp5 > 0, tmp3_m2 / tmp5, 0)
    tl.store(out + row * D + tl.cdiv(D, RBLOCK) + block_idx, tmp4, mask=block_idx < S)

    # wait for all updates to propagate
    buf4 = tl.load(primals_2 + row, mask=block_idx < S) - buf4
    # compute fused with intermediate
    tmp2 = buf4 * buf4 * tmp4
    tmp10 = tl.load(primals_1 + row, mask=block_idx < S)
    tmpa = tl.load(primals_2 + row, mask=block_idx < S)  # out_ptr2
    tmp6 = tl.load(primals_3 + row, mask=block_idx < S)
    tmp0 = tmpa - buf4 * tmp10
    tmp7 = tmp6
    tl.store(tmp7 + block_idx, tmp0, mask=block_idx < D)
    tmp1 = tmp0 * tmp0 * tmp4.to(tmp0.dtype.element_ty)
    tmpb = tl.load(primals_2 + row, mask=block_idx < S)  # out_ptr2
    tl.store(tmpb + block_idx, tmp1 + tmp2, mask=block_idx < D)

def fused_native_layer_norm(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6,
        device: torch.device = torch.device("cuda"), dtype: torch.dtype = torch.float16):
    if x.stride(0) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    M, D = x.shape

    # Below values are for WARP SIZE = 32 and can be different (24, 48, 96, 192, 384, 768)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    RBLOCK = min(max(triton.next_power_of_2(D), 4), MAX_FUSED_SIZE)
    assert D % RBLOCK == 0, "N should be multiplication of RBLOCK size"

    # create auxiliary tensors for Red
    buf0 = torch.empty([M, D // RBLOCK + 1], device=device, dtype=dtype)
    buf3 = torch.empty([M, D // RBLOCK + 1], device=device, dtype=dtype)
    out = torch.empty([M, D], device=device, dtype=dtype)
    tmp5 = torch.empty([M, D // RBLOCK + 1], device=device, dtype=torch.float32)

    num_warps = 4 if RBLOCK >= 2048 else 2
    grid = (x.shape[0],)

    # enqueue kernel
    with torch.cuda.device(x.device.index):
        triton_red_fused_native_layer_norm_0[grid](
            weight,

            x, weight, buf0,

            M, D,
            RBLOCK=RBLOCK,

            num_warps=num_warps,
            num_stages=1,
        )
        triton_red_fused_native_layer_norm_0[grid](
            weight,

            x, weight,

            M, D,
            RBLOCK=RBLOCK,

            num_warps=num_warps,
            num_stages=1,
        )
    # normalize
    return out
