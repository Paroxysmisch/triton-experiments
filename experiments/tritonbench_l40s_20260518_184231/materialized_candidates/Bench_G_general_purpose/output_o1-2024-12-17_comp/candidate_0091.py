import triton
import triton.language as tl
import torch

# ---------------------------
# Triton Kernel for MatMul
# ---------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, scales_ptr, zeros_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scales_n, stride_zeros_n,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # If using split-k, then pid_k = tl.program_id(2)
    # but for simplicity we only handle partial blocking here
    # though logic can be extended if SPLIT_K>1
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create a pointer for result accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # We loop over K dimension in chunks of BLOCK_SIZE_K
    # Convert from int4 in B for each chunk
    # Each int4 group is stored in an int32 (8 int4s per int32)
    # We'll handle the dequant logic using B_ptr, scales_ptr, zeros_ptr
    # at each step.

    # Range(0, K, BLOCK_SIZE_K)
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        # K size for the block
        k_offset = tl.arange(0, BLOCK_SIZE_K)
        # Offsets for A
        a_offset = (off_m[:, None] * stride_am) + ((k_block_start + k_offset[None, :]) * stride_ak)
        # Load A
        a = tl.load(A_ptr + a_offset, mask=(off_m[:, None] < M) & ((k_block_start + k_offset[None, :]) < K), other=0.0).to(tl.float32)

        # Offsets for B
        # We store B in int *B_ptr, but each int has 8 int4 values
        # so the global column offset is off_n
        # we need to figure out which elements off_n touches in B
        # We'll do bit manipulations to grab exact int4 from the int32
        b_offset = (k_block_start + k_offset[:, None]) * stride_bk + (off_n[None, :] * stride_bn)
        # indices in the flattened B array (in int32). Each block of 8 int4 is 1 int32, so offset is /8
        b_offset_int = b_offset // 8
        # position of the int4 in that int32
        b_offset_mod = b_offset % 8

        # Load the scales and zeros for B
        # Assume scales_ptr and zeros_ptr have dimension = N (stride_scales_n, stride_zeros_n)
        # We'll do it for each off_n
        scale = tl.load(scales_ptr + off_n, mask=(off_n < N), other=1.0)
        zp = tl.load(zeros_ptr + off_n,  mask=(off_n < N), other=0.0)

        # Load the int32 containing 8 int4
        b_int32 = tl.load(B_ptr + b_offset_int, mask=((k_block_start + k_offset[:, None]) < K) & (off_n[None, :] < N), other=0).to(tl.int32)

        # Extract the actual int4 value
        # Each int4 is 4 bits, we can shift and mask
        shift_amount = b_offset_mod * 4
        # get the 4-bit segment
        b_val_i32 = (b_int32 >> shift_amount) & 0xF
        # convert to float32
        b_val = b_val_i32.to(tl.float32)
        # dequant
        b_val = (b_val - zp[None, :]) * scale[None, :]

        # Multiply
        acc += tl.dot(a, b_val)

    # Now we store results
    # For SPLIT_K>1 we might need an atomic add, but we assume SPLIT_K=1 in this code
    # We'll directly store
    c_offset = off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    # Store result
    mask_c = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(C_ptr + c_offset, acc, mask=mask_c)


def matmul_dequantize_int4_s2(A, B_int4, B_scales, B_zeros):
    """
    A:        [M, K] float32
    B_int4:   [K*N//8] int32 (each int stores 8 int4)
    B_scales: [N] float32
    B_zeros:  [N] float32
    """
    M, K = A.shape
    N = B_scales.shape[0]
    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = lambda META: (
        (M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'],
        (N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N'],
    )

    matmul_kernel[grid](
        A, B_int4, B_scales, B_zeros, C,
        M, N, K,
        A.stride(0), A.stride(1),
        1, 1,  # B stride for K and N in this layout
        B_scales.stride(0), B_zeros.stride(0),
        C.stride(0), C.stride(1),
    )
    return C


def quantize_int4(weight):
    """
    weight: [K, N] float32
    Returns:
      packed_int4: torch.int32  [K*N//8]
      scales:       torch.float32 [N]
      zeros:        torch.float32 [N]
    """
    K, N = weight.shape
    weight_t = weight.view(K, N)

    # Compute scale/zero for each column
    # For example, min-max dynamic range
    mins = weight_t.min(dim=0)[0]
    maxs = weight_t.max(dim=0)[0]
    scales = (maxs - mins) / 15.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    zeros = (mins / scales).round()  # This is the "zero point"

    # Prepare int4 data
    # For each weight element:
    # val_int4 = round((val / scale) + zero)
    # then clamp between 0..15
    quant_mat = torch.round(weight_t / scales.unsqueeze(0) + zeros.unsqueeze(0))
    quant_mat.clamp_(0, 15)
    quant_mat_int = quant_mat.to(torch.int32)

    # Now pack 8 int4 in 1 int32
    # Flatten by row-major
    flatten_int = quant_mat_int.flatten()
    # We'll pad if needed to multiple of 8
    pad_len = (8 - (flatten_int.shape[0] % 8)) % 8
    if pad_len > 0:
        flatten_int = torch.cat([flatten_int, torch.zeros(pad_len, dtype=torch.int32, device=flatten_int.device)], dim=0)

    # Reshape to [num_ints], each holds 8 int4
    pack_size = flatten_int.shape[0] // 8
    flatten_int = flatten_int.view(pack_size, 8)

    shift_amounts = torch.arange(8, device=flatten_int.device, dtype=torch.int32) * 4
    # shift and accumulate
    packed_int4 = (flatten_int << shift_amounts).sum(dim=1)

    return packed_int4, scales, zeros


def unpack_int4(packed_int4, scales, zeros, K, N):
    """
    For testing or debugging. Unpacks the int4 matrix back to float32, shape [K, N].
    """
    num_elements = K * N
    pad_len = (8 - (num_elements % 8)) % 8
    total_len = num_elements + pad_len
    pack_size = total_len // 8

    # Expand each int32 into 8 int4
    # shape [pack_size]
    shifted = torch.stack([(packed_int4 >> (4*i)) & 0xF for i in range(8)], dim=1)  # [pack_size, 8]
    shifted = shifted.flatten()[:num_elements]

    # Dequant
    # We know that each column j in [K, N] has scale[j] and zero[j]
    # let's reconstruct
    out = shifted.view(K, N).float()
    col_idx = torch.arange(N, device=out.device).unsqueeze(0).expand(K, N)
    out = (out - zeros[col_idx]) * scales[col_idx]
    return out.view(K, N)
