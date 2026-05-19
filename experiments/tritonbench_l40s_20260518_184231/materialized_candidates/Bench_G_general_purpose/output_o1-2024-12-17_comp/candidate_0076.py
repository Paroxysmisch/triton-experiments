import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=1),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=4),
    ],
    key=['M', 'K'],
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr,  # float32 input
    a_ptr,    # int8 output
    as_ptr,   # float32 output (scale per row)
    M, K,
    stride_fpa_row, stride_fpa_col,
    stride_a_row, stride_a_col,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs
    row_id = tl.program_id(0)
    row_start = row_id * BLOCK_M
    # Offsets for rows
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    # Each row offset must be within M
    mask_m = row_offsets < M

    # We'll iterate over the entire row in steps of BLOCK_K
    max_val = tl.zeros([BLOCK_M], dtype=tl.float32)

    # For K-dimension iteration
    for col_start in range(0, K, BLOCK_K):
        # Column offsets
        col_offsets = col_start + tl.arange(0, BLOCK_K)
        mask_k = col_offsets < K
        fpa_ptrs = fpa_ptr + row_offsets[:, None] * stride_fpa_row + col_offsets[None, :] * stride_fpa_col
        mask = mask_m[:, None] & mask_k[None, :]
        block_vals = tl.where(mask, tl.load(fpa_ptrs, mask=mask, other=0.0), 0.0)
        abs_block_vals = tl.abs(block_vals)
        block_max = tl.maximum(tl.max(abs_block_vals, 1), max_val)
        max_val = block_max

    # Write out scale factor (max_abs / 127.0)
    scales = max_val / 127.0
    as_ptrs = as_ptr + row_offsets * 1
    tl.store(as_ptrs, scales, mask=mask_m)

    # Re-iterate to do the quantization
    for col_start in range(0, K, BLOCK_K):
        col_offsets = col_start + tl.arange(0, BLOCK_K)
        mask_k = col_offsets < K
        fpa_ptrs = fpa_ptr + row_offsets[:, None] * stride_fpa_row + col_offsets[None, :] * stride_fpa_col
        a_ptrs = a_ptr + row_offsets[:, None] * stride_a_row + col_offsets[None, :] * stride_a_col
        mask = mask_m[:, None] & mask_k[None, :]

        block_vals = tl.where(mask, tl.load(fpa_ptrs, mask=mask, other=0.0), 0.0)
        scale_vals = tl.broadcast_to(scales, [BLOCK_M])
        scaled_vals = block_vals / scale_vals[:, None]
        clamped_vals = tl.minimum(tl.maximum(scaled_vals, -127.0), 127.0)
        int8_vals = tl.cast(tl.round(clamped_vals), tl.int8)
        tl.store(a_ptrs, int8_vals, mask=mask)


def quantize_int8_perrow(fpa: torch.Tensor, block_m: int = 128, block_k: int = 128):
    M, K = fpa.shape
    fpa_contig = fpa.contiguous()
    a = torch.empty_like(fpa_contig, dtype=torch.int8)
    as_ = torch.empty(M, dtype=torch.float32, device=fpa.device)

    grid = ( (M + block_m - 1) // block_m, )
    quantize_int8_perrow_kernel[grid](
        fpa_contig, a, as_,
        M, K,
        fpa_contig.stride(0), fpa_contig.stride(1),
        a.stride(0), a.stride(1),
        BLOCK_M=block_m, BLOCK_K=block_k
    )
    return a, as_


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    AS_ptr, BS_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # For SPLIT-K
    k_offset = pid_k * BLOCK_SIZE_K

    a_scale_ptrs = AS_ptr + off_m
    b_scale_ptrs = BS_ptr + off_n
    a_scales = tl.load(a_scale_ptrs, mask=off_m < M, other=0.0)
    b_scales = tl.load(b_scale_ptrs, mask=off_n < N, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in steps
    for k_iter in range(k_offset, k_offset + BLOCK_SIZE_K, BLOCK_SIZE_K):
        # Clamped to the real K
        k_valid = tl.max_contiguous(tl.min(BLOCK_SIZE_K, K - k_iter), BLOCK_SIZE_K)
        if k_iter >= K:
            break

        for kk in range(0, k_valid, BLOCK_SIZE_K):
            current_k = k_iter + kk
            off_k = current_k + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = A_ptr + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
            b_ptrs = B_ptr + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)

            a_mask = (off_m[:, None] < M) & (off_k[None, :] < K)
            b_mask = (off_k[:, None] < K) & (off_n[None, :] < N)

            a_vals_i8 = tl.where(a_mask, tl.load(a_ptrs, mask=a_mask, other=0), 0)
            b_vals_i8 = tl.where(b_mask, tl.load(b_ptrs, mask=b_mask, other=0), 0)
            a_vals = tl.cast(a_vals_i8, tl.float32)
            b_vals = tl.cast(b_vals_i8, tl.float32)

            # Apply scales
            a_vals = a_vals * (1.0 / tl.broadcast_to(a_scales, [BLOCK_SIZE_M])[:, None])
            b_vals = b_vals * (1.0 / tl.broadcast_to(b_scales, [BLOCK_SIZE_N])[None, :])
            accumulator += tl.dot(a_vals, b_vals)

    off_C = C_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    c_mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(off_C, accumulator, mask=c_mask)


def matmul_int8(a: torch.Tensor, b: torch.Tensor, as_: torch.Tensor, bs_: torch.Tensor,
                block_m=128, block_n=128, block_k=32, split_k=1, out=None):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible dimensions"
    if out is None:
        out = torch.empty((M, N), dtype=torch.float32, device=a.device)

    grid = (
        (M + block_m - 1) // block_m,
        (N + block_n - 1) // block_n,
        split_k
    )
    matmul_kernel[grid](
        a, b, out,
        as_, bs_,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k, SPLIT_K=split_k
    )
    return out


def matmul_quantize_int8(fpa: torch.Tensor, fpb: torch.Tensor,
                         block_m=128, block_n=128, block_k=32, split_k=1):
    # Quantize per-row for fpa and fpb
    a, as_ = quantize_int8_perrow(fpa, block_m, block_k)
    b, bs_ = quantize_int8_perrow(fpb, block_k, block_n)
    # Matrix multiplication using quantized data
    return matmul_int8(a, b, as_, bs_, block_m, block_n, block_k, split_k)


def quantize_int8(weights: torch.Tensor, axis=0):
    # Per-row or per-column
    if axis == 0:
        max_abs = weights.abs().max(dim=1, keepdim=True)[0]
    else:
        max_abs = weights.abs().max(dim=0, keepdim=True)[0]

    scale = max_abs / 127.0
    scale = torch.clamp(scale, min=1e-8)  # avoid div by zero
    quantized = torch.round(weights / scale).clamp(-127, 127).to(torch.int8)

    return quantized, scale.squeeze()
