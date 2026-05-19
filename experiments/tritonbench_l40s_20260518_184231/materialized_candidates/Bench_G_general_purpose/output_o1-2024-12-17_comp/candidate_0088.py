import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------

@triton.jit
def matmul4_kernel(
    A_PTR, B_PTR, C_PTR,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_idxs = tl.arange(0, BLOCK_SIZE_K)

    mask_m = rm < M
    mask_n = rn < N

    # Create accumulators
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K in steps of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # Fetch A
        a_ptrs = A_PTR + (rm[:, None] * stride_am + (k + k_idxs)[None, :] * stride_ak)
        a = tl.where(
            (mask_m[:, None] & (k + k_idxs)[None, :] < K),
            tl.load(a_ptrs, mask=mask_m[:, None]),
            0.0
        )

        # Fetch B in int32
        b_ptrs = B_PTR + ((k + k_idxs)[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_vals = tl.where(
            ((k + k_idxs)[:, None] < K) & mask_n[None, :],
            tl.load(b_ptrs, mask=((k + k_idxs)[:, None] < K) & mask_n[None, :], other=0),
            0
        )

        # Dequantize scales and zeros
        scales_ptr_k = scales_ptr + (k + k_idxs) * stride_scales
        zeros_ptr_k = zeros_ptr + (k + k_idxs) * stride_zeros
        s = tl.load(scales_ptr_k)
        z = tl.load(zeros_ptr_k)

        # Dequantize B
        # Each int32 b_vals contains 8 int4. We'll process block_size_k * block_size_n.
        b_deq = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for offset in range(8):
            shift = (offset * 4)
            # Extract 4 bits
            val_4bit = (b_vals >> shift) & 0xF
            # Convert to float
            val_f32 = (val_4bit.to(tl.float32) - z[:, None]) * s[:, None]
            b_deq += val_f32 * ((rn[None, :] % 8) == offset)

        # Compute multiplication
        accum += tl.dot(a.to(tl.float32), b_deq)

    # Write back
    c_ptrs = C_PTR + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c = accum.to(tl.float16)
    tl.store(c_ptrs, c, mask=(mask_m[:, None] & mask_n[None, :]))


def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, out=None):
    M, K = x.shape
    K2, N = qweight.shape
    assert K == K2
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (
        ( (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M ),
        ( (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N )
    )

    if out is None:
        out = torch.empty((M, N), device=x.device, dtype=torch.float16)

    triton.run(matmul4_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        args=[
            x, qweight, out,
            scales, qzeros,
            M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            out.stride(0), out.stride(1),
            scales.stride(0), qzeros.stride(0)
        ],
        constants={
            'BLOCK_SIZE_M': BLOCK_SIZE_M,
            'BLOCK_SIZE_N': BLOCK_SIZE_N,
            'BLOCK_SIZE_K': BLOCK_SIZE_K
        }
    )
    return out

@triton.jit
def matmul_kernel(
    A_PTR, B_PTR, C_PTR,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_nk = tl.program_id(1)
    pid_n = pid_nk % ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    pid_k = pid_nk // ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    # Partial K-block
    KBlockSize = (K + SPLIT_K - 1) // SPLIT_K
    KStart = pid_k * KBlockSize
    KEnd = tl.minimum(KStart + KBlockSize, K)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_idxs = tl.arange(0, BLOCK_SIZE_K)

    mask_m = rm < M
    mask_n = rn < N

    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(KStart, KEnd, BLOCK_SIZE_K):
        a_ptrs = A_PTR + (rm[:, None] * stride_am + (k + k_idxs)[None, :] * stride_ak)
        a = tl.where(
            (mask_m[:, None] & (k + k_idxs)[None, :] < K),
            tl.load(a_ptrs, mask=mask_m[:, None]),
            0.0
        )

        b_ptrs = B_PTR + ((k + k_idxs)[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_vals = tl.where(
            ((k + k_idxs)[:, None] < K) & mask_n[None, :],
            tl.load(b_ptrs, mask=((k + k_idxs)[:, None] < K) & mask_n[None, :], other=0),
            0
        )

        scales_ptr_k = scales_ptr + (k + k_idxs) * stride_scales
        zeros_ptr_k = zeros_ptr + (k + k_idxs) * stride_zeros
        s = tl.load(scales_ptr_k)
        z = tl.load(zeros_ptr_k)

        b_deq = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for offset in range(8):
            shift = (offset * 4)
            val_4bit = (b_vals >> shift) & 0xF
            val_f32 = (val_4bit.to(tl.float32) - z[:, None]) * s[:, None]
            b_deq += val_f32 * ((rn[None, :] % 8) == offset)

        accum += tl.dot(a.to(tl.float32), b_deq)

    rm_mask = rm < M
    rn_mask = rn < N

    c_ptrs = C_PTR + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    # Atomic accumulation if SPLIT_K > 1
    if SPLIT_K > 1:
        old_val = tl.where(rm_mask[:, None] & rn_mask[None, :], tl.load(c_ptrs), 0.0)
        new_val = old_val + accum
        tl.store(c_ptrs, new_val, mask=rm_mask[:, None] & rn_mask[None, :])
    else:
        tl.store(c_ptrs, accum, mask=rm_mask[:, None] & rn_mask[None, :])

def matmul_dequantize_int4_s2(x, qweight, scales, qzeros, split_k=1):
    M, K = x.shape
    K2, N = qweight.shape
    assert K == K2
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    if split_k < 1:
        split_k = 1

    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)

    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n * split_k)

    triton.run(matmul_kernel,
               grid=grid,
               num_warps=4,
               num_stages=2,
               args=[
                   x, qweight, out,
                   scales, qzeros,
                   M, N, K,
                   x.stride(0), x.stride(1),
                   qweight.stride(0), qweight.stride(1),
                   out.stride(0), out.stride(1),
                   scales.stride(0), qzeros.stride(0),
                   split_k
               ],
               constants={
                   'BLOCK_SIZE_M': BLOCK_SIZE_M,
                   'BLOCK_SIZE_N': BLOCK_SIZE_N,
                   'BLOCK_SIZE_K': BLOCK_SIZE_K
               })

    return out.half()

@triton.jit
def dequantize_kernel(
    B_PTR, OUT_PTR, scales_ptr, zeros_ptr,
    K, N,
    stride_bk, stride_bn,
    stride_outk, stride_outn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_k = rk < K
    mask_n = rn < N

    b_ptrs = B_PTR + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    b_vals = tl.where(
        mask_k[:, None] & mask_n[None, :],
        tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0),
        0
    )

    s_ptrs = scales_ptr + rk * stride_scales
    z_ptrs = zeros_ptr + rk * stride_zeros

    s = tl.load(s_ptrs, mask=mask_k)
    z = tl.load(z_ptrs, mask=mask_k)

    out = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
    for offset in range(8):
        shift = (offset * 4)
        val_4bit = (b_vals >> shift) & 0xF
        val_f32 = (val_4bit.to(tl.float32) - z[:, None]) * s[:, None]
        out_part = val_f32.to(tl.float16) * ((rn[None, :] % 8) == offset)
        out += out_part

    out_ptrs = OUT_PTR + (rk[:, None] * stride_outk + rn[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=mask_k[:, None] & mask_n[None, :])

def dequantize_int4(qweight, scales, qzeros):
    K, N = qweight.shape
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    grid = (
        ( (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K ),
        ( (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N )
    )

    out = torch.empty((K, N), device=qweight.device, dtype=torch.float16)

    triton.run(dequantize_kernel,
               grid=grid,
               num_warps=4,
               num_stages=2,
               args=[
                   qweight, out,
                   scales, qzeros,
                   K, N,
                   qweight.stride(0), qweight.stride(1),
                   out.stride(0), out.stride(1),
                   scales.stride(0), qzeros.stride(0)
               ],
               constants={
                   'BLOCK_SIZE_K': BLOCK_SIZE_K,
                   'BLOCK_SIZE_N': BLOCK_SIZE_N
               })
    return out

def matmul_dequantize_int4_s1(x, qweight, scales, qzeros):
    # Dequantize weight
    dequant_w = dequantize_int4(qweight, scales, qzeros)
    # Use PyTorch mm
    return torch.mm(x.half(), dequant_w)
