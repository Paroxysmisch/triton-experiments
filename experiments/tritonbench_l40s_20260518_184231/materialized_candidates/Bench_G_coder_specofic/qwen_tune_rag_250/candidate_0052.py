_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for i in range(start_iter, end_iter, BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    acc = tl.dot(locks, acc.to(A.dtype))
    C = C + (ram[:, None] * stride_cm + rbn[None, :] * stride_cn)
    tl.store(C, acc.to(C.dtype.element_ty))

@triton.jit()
def first_wave(A, B, C,
               M, N, K,
               locks,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               total_full_tiles_streamk, total_partial_tiles_streamk,
               iters_per_tile,
               start_iter, end_iter,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):

    # for the first wave, we only launch full tiles, this way we avoid having to initialize the accumulator with random data.
    # initialization is slow and generates a lot of DRAM traffic

    # where are we in the grid
    tile_id = start_iter // iters_per_tile
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    tile_id_in_group = tile_id % total_full_tiles_streamk
    num_tiles_in_group = tl.minimum(total_partial_tiles_streamk, total_full_tiles_streamk)

    # for the full tiles, we avoid random access to the L1$ by always loading the same two chunks of A and B in each program.
    rk_base = tile_id_in_group * 2 * BLOCK_K
    # mask last partial tile
    tile_mask = tl.where(tile_id_in_group < num_tiles_in_group - 1, 1, 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = rk_base + tl.arange(0, BLOCK_K)
    # pointers
    A0 = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B0 = B + (rk[None, :] * stride_bk + rbn[None, :] * stride_bn)
    A1 = A + (ram[:, None] * stride_am + (rk + BLOCK_K)[None, :] * stride_ak)
    B1 = B + ((rk + BLOCK_K)[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for i in range(start_iter, end_iter, 2 * BLOCK_K):
        a0 = tl.load(A0, eviction_policy='evict_last')
        b0 = tl.load(B0, eviction_policy='evict_last')
        a1 = tl.load(A1, eviction_policy='evict_last')
        b1 = tl.load(B1, eviction_policy='evict_last')
        acc0 += tl.dot(a0, b0)
        acc1 += tl.dot(a1, b1)
        A0 += 2 * BLOCK_K * stride_ak
        B0 += 2 * BLOCK_K * stride_bk
        A1 += 2 * BLOCK_K * stride_ak
        B1 += 2 * BLOCK_K * stride_bk

    acc0 = tl.dot(locks, acc0.to(A.dtype))
    acc1 = tl.dot(locks, acc1.to(A.dtype))
    acc0 = tl.where(tile_mask[:, None], acc0, 0)
    acc1 = tl.where(tile_mask[:, None], acc1, 0)
    acc = acc0 + acc1
    # write back
    C = C + (ram[:, None] * stride_cm + rbn[None, :] * stride_cn)
    tl.store(C, acc.to(C.dtype.element_ty))

@triton.jit()
def full_tiles(A, B, C,
               M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               total_tiles_streamk,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):

    pid = tl.program_id(axis=0)

    tile_id = pid * 2  # because we launch 2 tiles per program
    # we only want to launch full tiles, the rest are handled by the "partial tiles" kernel
    tile_id_in_group = tile_id % total_tiles_streamk
    num_tiles_in_group = total_tiles_streamk

    # the last program may have less tiles
    tile_mask = tile_id_in_group < num_tiles_in_group

    # swizzle tile id to improve L2 performance
    pid_m = tile_id // (num_tiles_in_group // GROUP_M)
    pid_n = tile_id % (num_tiles_in_group // GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for i in range(0, 2 * BLOCK_K, BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    acc = tl.dot(locks, acc.to(A.dtype))
    acc = tl.where(tile_mask[:, None], acc, 0)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc.to(C.dtype.element_ty))

class matmul(torch.autograd.Function):

    @staticmethod
    def _call(a, b, trans_a, trans_b, trans_c, M, N, K, BLK_M, BLK_N, BLK_K, num_stages, num_warps, groups, locks, args, K_multiple):
        if K_multiple is None:
            K_multiple = triton.next_power_of_2(K)

        assert A.shape[2] == B.shape[1], "incompatible dimensions"
        assert K % BLK_K == 0, "K must be divisible by BLK_K"

        if not trans_c:
            c = torch.empty((groups, M, N), device=a.device, dtype=a.dtype)
        else:
            c = torch.empty((groups, N, M), device=a.device, dtype=a.dtype)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

        tiles_per_program = 2  # two tiles per program, except in the last program
        total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
        total_tiles_streamk = total_tiles - (total_tiles % tiles_per_program)

        full_tiles_per_program = total_tiles_streamk // tiles_per_program
        total_partial_tiles = total_tiles - full_tiles_per_program
        assert total_partial_tiles < tiles_per_program, "bug in the tile counting"

        with torch.cuda.device(a.device.index):
            if total_partial_tiles > 0:
                # partial tiles cannot be swizzled
                partial_tiles_per_program = total_partial_tiles
                num_partial_programs = triton.cdiv(total_partial_tiles, partial_tiles_per_program)
                first_wave[grid](a, b, c,
                                 M, N, K,
                                 locks,
                                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                                 full_tiles_per_program,  # start iter
                                 full_tiles_per_program + partial_tiles_per_program,  # end iter
                                 BLOCK_M: BLK_M, BLOCK_N: BLK_N, BLOCK_K: BLK_K,
                                 ACC
