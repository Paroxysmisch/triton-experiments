scales_ptr, zeros_ptr, g_ptr,
    M, N, K,
    bits, maxq,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Triton kernel for transposed matrix multiplication
    infearure_per_bits = 32 // bits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M)
    b_ptrs = b_ptr + (
        (offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn
    )
    g_ptrs = g_ptr + offs_k
    g_idx = tl.load(g_ptrs)

    scales_ptrs = scales_ptr + offs_bn[None, :] + g_idx[:, None] * stride_scales
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)
        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)
        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = (zeros + 1)

        a = tl.load(a_ptrs, mask=a_mask, other=0.)
        b = tl.load(b_ptrs)

        b = (b >> shifter[:, None]) & maxq
        b = (b - zeros) * scales
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def quant_matmul_248(
    a: torch.Tensor, b: torch.Tensor, scales: torch.Tensor,
    zeros: torch.Tensor, g: torch.Tensor, bits: int, maxq: int
):
    # Wrapper function for quant_matmul_248_kernel
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(a.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(b.shape[1], META["BLOCK_SIZE_N"]),
    )
    quant_matmul_248_kernel[grid](
        a, b, c,
        scales, zeros, g,
        a.shape[0], b.shape[1], a.shape[1],
        bits, maxq,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), zeros.stride(0),
    )
    return c


def transpose_quant_matmul_248(
    a: torch.Tensor, b: torch.Tensor, scales: torch.Tensor,
    zeros: torch.Tensor, g: torch.Tensor, bits: int, maxq: int
):
    # Wrapper function for transpose_quant_matmul_248_kernel
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(a.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(b.shape[1], META["BLOCK_SIZE_N"]),
    )
    transpose_quant_matmul_248_kernel[grid](
        a, b, c,
        scales, zeros, g,
        a.shape[0], b.shape[1], a.shape[1],
        bits, maxq,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), zeros.stride(0),
    )
    return c
