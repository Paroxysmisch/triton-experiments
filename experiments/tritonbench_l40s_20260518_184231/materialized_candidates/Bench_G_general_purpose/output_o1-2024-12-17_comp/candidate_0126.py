import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config(meta={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
        triton.Config(meta={"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
        triton.Config(meta={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Create a pointer for output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_SIZE_K
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak

        # Load A
        a_vals = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0.0)

        # Each column in B is 4-bit packed, so decode in partial
        # We process each tile of BLOCK_SIZE_N columns
        b_indices = offs_n[None, :] < N
        b_ptrs = b_ptr + (k_offsets[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        # Load scale/zero; group idx from (k_offsets // groupsize)
        group_idx = (k_offsets // groupsize)
        scale_ptrs = scales_ptr + group_idx[:, None] * stride_scales_g + offs_n[None, :] * stride_scales_n
        zero_ptrs = zeros_ptr + group_idx[:, None] * stride_zeros_g + offs_n[None, :] * stride_zeros_n

        # We decode int4 from each 32-bit chunk. All math in loop
        # For each k in [k_start, k_start+BLOCK_SIZE_K)
        b_acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(BLOCK_SIZE_K):
            # load scales, zeros
            s = tl.load(scale_ptrs[kk, :], mask=b_indices, other=0.0)
            z = tl.load(zero_ptrs[kk, :], mask=b_indices, other=0.0)
            packed_val = tl.load(b_ptrs[kk, :], mask=b_indices, other=0).to(tl.int32)
            # extract 4-bit from packed_val
            # offset is (k_start + kk) mod 2 nibble half if necessary
            # Actually each int32 holds 8 int4 values along the 'N' dim, but for simplicity,
            # assume 1 int4 in each int32. Adjust as needed for real use-cases.
            val = packed_val & 0xF
            # dequant
            val_fp = s * (val.to(tl.float32) - z)
            b_acc[kk, :] = val_fp

        # Multiply
        # a_vals shape [BLOCK_SIZE_M, BLOCK_SIZE_K], b_acc shape [BLOCK_SIZE_K, BLOCK_SIZE_N]
        tmp = tl.dot(a_vals.to(tl.float32), b_acc)
        acc += tmp

    # Store
    c_out = acc.to(tl.float16)
    tl.store(c_ptrs, c_out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config(meta={"BLOCK_SIZE_K": 64, "BLOCK_SIZE_N": 64}),
        triton.Config(meta={"BLOCK_SIZE_K": 128, "BLOCK_SIZE_N": 64}),
        triton.Config(meta={"BLOCK_SIZE_K": 64, "BLOCK_SIZE_N": 128}),
    ],
    key=["K", "N"],
)
@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    fpb_ptrs = fpb_ptr + offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn

    # Prepare accumulators
    out_val = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K in tile
    for kk in range(BLOCK_SIZE_K):
        k_idx = offs_k[kk]
        if k_idx < K:
            group_idx = k_idx // group_size
            scales_ptr = b_scale_ptr + group_idx * stride_bsk + offs_n * stride_bsn
            zeros_ptr = b_zp_ptr + group_idx * stride_bzpk + offs_n * stride_bzpn
            # load scale/zero
            s = tl.load(scales_ptr, mask=(offs_n < N), other=0.0)
            z = tl.load(zeros_ptr, mask=(offs_n < N), other=0.0)

            # load packed int4
            b_val_ptr = b_ptr + k_idx * stride_bk + offs_n * stride_bn
            b_val_packed = tl.load(b_val_ptr, mask=(offs_n < N), other=0).to(tl.int32)
            # decode
            val = b_val_packed & 0xF
            val_fp = s * (val.to(tl.float32) - z)
            out_val[kk, :] = val_fp

    # Store
    tl.store(fpb_ptrs, out_val, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))


def dequantize_int4(b, b_scale, b_zero, K, N, group_size):
    out = torch.empty((K, N), dtype=torch.float32, device=b.device)
    grid = lambda META: ( (K + META["BLOCK_SIZE_K"] - 1) // META["BLOCK_SIZE_K"],
                          (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"] )
    dequantize_kernel[grid](
        b, b_scale, b_zero, out,
        K, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero.stride(0), b_zero.stride(1),
        out.stride(0), out.stride(1)
    )
    return out


def matmul_dequantize_int4_s1(a, b, b_scale, b_zero, groupsize):
    K, N = b.shape
    M = a.shape[0]
    out = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda META: ( (M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"],
                          (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"] )
    matmul4_kernel[grid](
        a, b, out,
        b_scale, b_zero,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero.stride(0), b_zero.stride(1),
        groupsize, 0
    )
    return out


def quantize_int4(weight: torch.Tensor, group_size: int):
    # shape = (K, N)
    K, N = weight.shape
    # group-based stats
    groups = (K + group_size - 1) // group_size
    scales = torch.zeros((groups, N), device=weight.device, dtype=torch.float32)
    zeros = torch.zeros((groups, N), device=weight.device, dtype=torch.float32)
    qweight = torch.zeros_like(weight, dtype=torch.int32)

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, K)
        w_slice = weight[start:end, :]  # (gs or less, N)
        mn = w_slice.min(dim=0)[0]  # per-col min
        mx = w_slice.max(dim=0)[0]  # per-col max
        scale = (mx - mn) / 15.0
        scale[scale == 0.0] = 1e-8
        zero = torch.round(mn / scale)
        scales[g, :] = scale
        zeros[g, :] = zero
        # quant
        tmp = torch.round(w_slice / scale - zero).clamp_(0, 15)
        # store
        qweight[start:end, :] = tmp.int()

    return qweight, scales, zeros, K, N, group_size
