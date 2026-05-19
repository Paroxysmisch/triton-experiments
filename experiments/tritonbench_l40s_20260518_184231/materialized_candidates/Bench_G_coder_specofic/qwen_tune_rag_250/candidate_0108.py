group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + ((offs_bk[:, None] // infearure_per_bits) * stride_bk + offs_n[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_bk
    g_idx = tl.load(g_ptrs)

    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
    zeros_ptrs = zeros_ptr + (offs_n[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

    shifter = (offs_bk % infearure_per_bits) * bits
    zeros_shifter = (offs_n % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, num_pid_n):
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = (zeros + 1)

        a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
        b = (b - zeros) * scales  # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N
        scales_ptrs += BLOCK_SIZE_N
        zeros_ptrs += BLOCK_SIZE_N // infearure_per_bits

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def w4a16_matmul(x, qweight, scales, qzeros, g_idx, bits, maxq):
    if bits == 4:
        return _w4a16_matmul(x, qweight, scales, qzeros, g_idx)
    elif bits == 16:
        return _w4a16_matmul(x, qweight, scales, qzeros, g_idx, use_int8=False)
    else:
        raise ValueError("Only bits=4 and bits=16 are supported")


def w4a16_matmul_t(x, qweight, scales, qzeros, g_idx, bits, maxq):
    if bits == 4:
        return _w4a16_matmul_t(x, qweight, scales, qzeros, g_idx)
    elif bits == 16:
        return _w4a16_matmul_t(x, qweight, scales, qzeros, g_idx, use_int8=False)
    else:
        raise ValueError("Only bits=4 and bits=16 are supported")


def _w4a16_matmul(x, qweight, scales, qzeros, g_idx, use_int8=True):
    M, N, K = x.shape, qweight.shape
    assert x.shape[-1] == (qweight.shape[-2] * 8), "Incompatible dimensions"

    c = torch.empty((M[0], N[1]), device=x.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M[0], META['BLOCK_SIZE_M']) * triton.cdiv(N[1], META['BLOCK_SIZE_N']), )
    matmul_248_kernel[grid](
        x, qweight, c, scales, qzeros, g_idx, M[0], N[1], K[1], 4 if use_int8 else 16, 128, x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1), c.stride(0), c.stride(1),
        scales.stride(0), qzeros.stride(0))
    return c


def _w4a16_matmul_t(x, qweight, scales, qzeros, g_idx, use_int8=True):
    M, N, K = x.shape, qweight.shape
    assert x.shape[-1] == (qweight.shape[-2] * 8), "Incompatible dimensions"

    c = torch.empty((M[0], N[1]), device=x.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M[0], META['BLOCK_SIZE_M']) * triton.cdiv(N[1], META['BLOCK_SIZE_N']), )
    transpose_matmul_248_kernel[grid](
        x, qweight, c, scales, qzeros, g_idx, M[0], N[1], K[1], 4 if use_int8 else 16, 128, x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1), c.stride(0), c.stride(1),
        scales.stride(0), qzeros.stride(0))
    return c
