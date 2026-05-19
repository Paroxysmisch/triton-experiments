import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul4_kernel(
    A_ptr, B_ptr, C_ptr,
    scales_ptr, zeros_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K, group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

    # Create pointers for A and C
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Block-based loop
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k_block_start
        # Guard
        k_mask = k_offsets < K

        # Load A
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)

        # B offset calculations
        b_offset = (k_offsets // group_size)
        # For each group, we use one scale/zero
        b_scale = tl.load(scales_ptr + b_offset, mask=k_mask, other=1.0)
        b_zero = tl.load(zeros_ptr + b_offset, mask=k_mask, other=0.0)
        b_int_ptrs = B_ptr + (k_offsets // 8) * stride_bk  # each int32 has 8 int4

        # Dequantize B
        b_vals = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(8):
            # shift: 4 bits for each int4
            shift = i * 4
            # load int32
            b_ints = tl.load(b_int_ptrs + (n_start + tl.arange(0, BLOCK_SIZE_N)) * stride_bn, mask=k_mask[:, None], other=0)
            extracted = (b_ints >> shift) & 0xF
            extracted_f = extracted.to(tl.float32)
            # apply scale/zero
            # each row in a block might correspond to single scale/zero or per-col 
            # but here we assume per-row for demonstration
            b_vals += tl.where(tl.arange(0, BLOCK_SIZE_K)[:, None] % 8 == i,
                               (extracted_f - b_zero[:, None]) * b_scale[:, None],
                               0.0)
            b_int_ptrs += (tl.arange(0, BLOCK_SIZE_N) * 0)  # no actual shift in pointer each loop iteration

        # Compute FMA
        acc += tl.dot(a.to(tl.float32), b_vals)

        # Advance pointers
        a_ptrs += (BLOCK_SIZE_K * stride_ak)
    # Store the results
    c = acc.to(tl.float16)
    mask_m = offs_m < M
    mask_n = offs_n < N
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def matmul_dequantize_int4_gptq(A, Bq, scales, zeros, group_size):
    """
    A: FP16 matrix of shape (M, K)
    Bq: packed int4 matrix in int32 form, shape roughly (K//8, N)
    scales: dequant scale array for B
    zeros: dequant zero array for B
    group_size: grouping parameter for GPTQ
    """
    assert A.dtype == torch.float16, "A must be float16"
    assert Bq.dtype == torch.int32, "Bq must be int32 (packed int4)"
    M, K = A.shape
    # Bq shape is tricky but let's assume it matches with N properly
    # N can be inferred from Bq's shape or external knowledge
    # For demonstration, assume we require user to pass N
    # We'll derive N from Bq if possible
    # Bq has shape [K//8, N], but let's get from Bq
    Kdiv8, N = Bq.shape
    assert Kdiv8 * 8 == K, "Bq shape mismatch with K"

    A_ptr = A
    B_ptr = Bq
    scales_ptr = scales
    zeros_ptr = zeros

    # Prepare output C
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    grid = lambda META: ( (M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'],
                          (N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N'] )

    matmul4_kernel[grid](
        A_ptr, B_ptr, C, 
        scales_ptr, zeros_ptr,
        A.stride(0), A.stride(1),
        Bq.stride(0), Bq.stride(1),
        C.stride(0), C.stride(1),
        M, N, K, group_size
    )
    return C


def quantize_int4(weights, group_size):
    """
    weights: a 2D FP16 or FP32 tensor (K, N)
    group_size: grouping for GPTQ quantization
    Returns (packed int4 tensor, scales, zeros) 
    """
    assert weights.dim() == 2, "weights must be 2D"
    K, N = weights.shape
    weights_f = weights.float()
    # For demonstration, quantize row-wise in groups of size 'group_size'
    # e.g. if group_size=32, each 32-row chunk shares the same scale/zero
    # We'll store one scale and zero per row for simplicity; 
    # actual GPTQ may differ in grouping strategy

    packed = torch.zeros((K // 8, N), dtype=torch.int32, device=weights.device)
    scales = torch.zeros((K,), dtype=torch.float32, device=weights.device)
    zeros = torch.zeros((K,), dtype=torch.float32, device=weights.device)

    for k in range(K):
        row = weights_f[k]
        min_val = row.min()
        max_val = row.max()
        # Compute scale & zero
        scale = (max_val - min_val) / 15.0 if max_val > min_val else 1.0
        zero = -min_val / scale if scale != 0.0 else 0.0
        scales[k] = scale
        zeros[k] = zero
        # Quantize row
        q_row = ((row * (1.0 / scale)) + zero).round().clamp(0, 15)
        q_row = q_row.to(torch.int32)
        # Pack int4 into int32
        num_el = q_row.numel()
        tmp_storage = []
        for n_idx in range(num_el):
            # each int32 can store 8 int4
            sub_idx = n_idx // 8
            shift = (n_idx % 8) * 4
            val = (q_row[n_idx] & 0xF) << shift
            if sub_idx >= len(tmp_storage):
                tmp_storage.append(val)
            else:
                tmp_storage[sub_idx] |= val
        for i, val in enumerate(tmp_storage):
            packed[k // 8 + i if (k % 8 == 0) else k // 8, :] = 0  # safe initialization
        # This naive approach is row-based; we need to place packed data per column
        # For simplicity, place the row's 32-bit segments across columns
        for col_idx in range(N):
            # each element in q_row is for col_idx
            # gather them in 8-element blocks
            sub_idx = col_idx // 8
        # For a real layout, consider rearranging for performance. Here is a simplified approach:
        # We'll do per-col packing in a direct manner:
        for col in range(N):
            idx_int32 = col
            bit_shift = (k % 8) * 4
            val = (q_row[col] & 0xF) << bit_shift
            packed[k // 8, idx_int32] |= val

    return packed, scales, zeros
