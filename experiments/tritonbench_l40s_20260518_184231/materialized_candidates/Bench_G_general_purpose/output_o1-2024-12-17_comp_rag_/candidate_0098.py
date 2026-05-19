import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton Kernel for INT4 Matrix Multiplication (on-the-fly dequantization)
# ------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "SPLIT_K": 1}, num_stages=2, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, scales_ptr, zps_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scales_k, stride_scales_n,
    stride_zps_k, stride_zps_n,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    """Computes C = A * B, where B is stored with INT4 quantization, on-the-fly dequantizing B."""
    pid_m = tl.program_id(0)  
    pid_n = tl.program_id(1)
    # Allow multi-k-split
    pid_k = tl.program_id(2)
    # Block of the output
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk_offset = pid_k * BLOCK_SIZE_K
    # Create accumulator in FP32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Loop over K
    for k in range(0, BLOCK_SIZE_K):
        k_idx = rk_offset + k
        # Load A
        a_offs = rm * stride_am + k_idx * stride_ak
        a_mask = (rm < M) & (k_idx < K)
        a_val = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)
        # Load INT4 B
        b_offs = k_idx * stride_bk + rn * stride_bn
        b_mask = (rn < N) & (k_idx < K)
        b_int = tl.load(b_ptr + b_offs, mask=b_mask, other=0)
        # Extract scale and zero point
        s_offs = k_idx * stride_scales_k + rn * stride_scales_n
        zp_offs = k_idx * stride_zps_k + rn * stride_zps_n
        scale_v = tl.load(scales_ptr + s_offs, mask=b_mask, other=0.0)
        zp_v = tl.load(zps_ptr + zp_offs, mask=b_mask, other=0)
        # Reconstruct INT4 in loop. 8 int4s packed in one int32.
        partial = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(8):
            # shift and mask out 4 bits
            b_4 = ((b_int << (28 - i * 4)) >> 28) & 0xF
            zp_4 = ((zp_v << (28 - i * 4)) >> 28) & 0xF
            # Convert scale
            b_deq = (b_4.to(tl.int32) - zp_4.to(tl.int32)).to(tl.float32) * scale_v
            # Outer product
            a_i = a_val
            partial += a_i[:, None] * b_deq[None, :]
        acc += partial
    # Write back result
    if SPLIT_K == 1:
        # No atomic add needed
        c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(c_ptr + c_offs, acc, mask=c_mask)
    else:
        # Use atomic add
        c_offs = rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        old_val = tl.load(c_ptr + c_offs, mask=c_mask, other=0.0)
        new_val = old_val + acc
        tl.store(c_ptr + c_offs, new_val, mask=c_mask)

def matmul_dequantize_int4_s2(a, b_int4, b_scales, b_zps, split_k=1, out=None):
    """
    Prepares and launches the matmul_kernel with int4-quantized B.
    a:      [M, K], float32
    b_int4: [K, N] in INT4 packed format (every row element has 8 int4)
    b_scales, b_zps: scaling and zero points
    split_k: how many splits along K dimension
    out:    [M, N], optional
    """
    assert a.dtype == torch.float32, "Matrix A must be float32."
    M, K_ = a.shape
    Kb, N = b_int4.shape
    assert K_ == Kb, "Incompatible dimensions for A and B."
    if out is None:
        out = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (
        (M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"],
        (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"],
        split_k
    )
    matmul_kernel[grid](
        a, b_int4, b_scales, b_zps, out,
        M, N, K_,
        a.stride(0), a.stride(1),
        b_int4.stride(0), b_int4.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        b_zps.stride(0), b_zps.stride(1),
        out.str
