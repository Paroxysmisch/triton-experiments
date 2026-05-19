import torch
import triton
import triton.language as tl


@triton.jit
def _lu_partial_pivoting(a, r, m, n, k, info, sgn, _2, _3,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # This function does both the LU decomposition and the partial pivoting.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    pid = pid_m + pid_n * BLOCK_SIZE_M
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    # now let's do the factorization!
    if pid < num_pid_m:
        # compute the row offset of the current program
        offs_r = (pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
        offs_k = min(k + pid_n * BLOCK_SIZE_N, n)
        # find the max in the column
        max_elem = 0
        max_idx = 0
        idx_j_start = tl.multiple_of(pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)
        idx_j_end = idx_j_start + BLOCK_SIZE_N
        for j in range(idx_j_start, idx_j_end, BLOCK_SIZE_N):
            j = tl.multiple_of(j, BLOCK_SIZE_N)
            a_j = tl.load(a + j + offs_r * n, mask=(offs_r < m) & (j < n), other=0.0)
            abs_a_j = tl.abs(a_j)
            max_elem_new = tl.maximum(max_elem, abs_a_j)
            # We use:
            #     ((x != 0) & (y != 0)) == (x * y != 0)
            # here to avoid masking load when `max_elem == abs_a_j`
            is_larger = (abs_a_j * max_elem_new != 0) & (abs_a_j > max_elem)
            max_idx = tl.where(is_larger, j, max_idx)
            max_elem = tl.where(is_larger, max_elem_new, max_elem)
        # now we have the max element, we need to record the corresponding row id
        # which will be used later to perform the actual pivoting outside this kernel
        r_offs = (pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) // BLOCK_SIZE_M
        tl.store(r + r_offs, max_idx // BLOCK_SIZE_N)
        max_idx = max_idx % BLOCK_SIZE_N
        # i is the current row index
        for i in range(0, BLOCK_SIZE_M):
            i = pid_m * BLOCK_SIZE_M + i
            if i < m:
                # get the offsets for the current row
                offs_i = i + offs_r * n
                # perform the pivoting: swap rows if needed
                cond_swap = (i >= max_idx) & (offs_r < m)
                offs_i, max_idx = tl.where(cond_swap, max_idx, offs_i), tl.where(cond_swap, i, max_idx)
                # load the row
                a_i = tl.load(a + offs_i, mask=offs_r < m, other=0)
                # divide the row by the pivot
                denom = tl.load(a + max_idx * n + i)
                # set the pivot position
                a_ii = tl.where(i == 0, 1.0, 0.0)
                a_i = a_i / denom
                a_i = tl.where((offs_r == i) & (offs_r < m), a_ii, a_i)
                # write back the result
                tl.store(a + offs_i, a_i, mask=offs_r < m)
                # accumulate the sign
                sgn = tl.where(cond_swap, -sgn, sgn)
        # save sign
        offs_sgn = pid * BLOCK_SIZE_M // BLOCK_SIZE_SGN_UNIT + tl.arange(0, BLOCK_SIZE_SGN_ACCUM)
        tl.store(sgn + offs_sgn, sgn)
    elif pid_m < num_pid_m:
        # perform the actual pivoting outside this kernel since it requires shared memory
        # which is way too slow!
        offs_r = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
        p_offs = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        # load the ids
        r_i = tl.load(r + offs_r)
        p = tl.load(p + p_offs)
        # apply the pivot
        tmp = tl.load(a + p * n + offs_r)
        tl.store(a + p * n + offs_r, tl.load(a + r_i * n + offs_r))
        tl.store(a + r_i * n + offs_r, tmp)
        # update the index
        tl.store(p + p_offs, r_i)
    # reset the pointer
    k = tl.multiple_of(k, BLOCK_SIZE_K)
    for i in range(k, n, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        a_i = tl.load(a + i + offs_r * n, mask=(offs_r < m) & (i < n), other=0.)
        denom = tl.load(a + i * n + i)
        # set the pivot position
        a_ii = tl.where((i == 0) | (offs_r == i), 1., 0.)
        a_i = tl.where((offs_r < m) & (i < n), a_ii, 0.)
        # write back the result
        tl.store(a + i + offs_r * n, a_i, mask=(offs_r < m) & (i < n))
    # check if we are done
    if pid_m == 0:
        # manually unroll the last iteration so we can save the sign
        i = n - BLOCK_SIZE_N
        i = tl.multiple_of(i, BLOCK_SIZE_N)
        a_i = tl.load(a + i + offs_r * n, mask=(offs_r < m) & (i < n), other=0.)
        denom = tl.load(a + i * n + i)
        # set the pivot position
        a_ii = tl.where((i == 0) | (offs_r == i), 1., 0.)
        a_i = tl.where((offs_r < m) & (i < n), a_ii, 0.)
        # write back the result
        tl.store(a + i + offs_r * n, a_i, mask=(offs_r < m) & (i < n))
        sgn = tl.math.copysign(tl.load(sgn + tl.arange(0, BLOCK_SIZE_SGN_ACCUM)), denom)
    # add info
    if pid == 0:
        tl.store(info, 0)
    # next loop iteration
    pid_m += num_pid_m
    pid = pid_m + pid_n * BLOCK_SIZE_M
    return pid.to(tl.int64), info, sgn


def _lu_with_pivoting(a, *, out_pivot=None, out_info=None, out_sgn=None):
    # This function calls the above kernel multiple times.
    # See https://github.com/openai/triton/issues/958 for more details about the approach taken below.
    # Get the problem size
    m, n = a.shape[-2:]
    # Figure out the number of blocks in x-dimension
    # and the number of blocks in y-dimension
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    # [2021-09-24] Using heuristics from the meta-benchmark, we set the number
    # of warps to 4 for all architectures. This is because the kernel is I/O bound
    # and bandwidth is likely the limiting factor given that we are transferring
    # 2 tiles between SMEM and DRAM per program instance.
    num_warps = 4
    # Compute the number of tiles in x- and y-direction
    num_tile_m = triton.cdiv(m, BLOCK_SIZE_M)
    num_tile_n = triton.cdiv(n, BLOCK_SIZE_N)
    # Allocate tensor to keep track of pivot indices
    if out_pivot is None:
        pivot = torch.empty((num_tile_m,), dtype=torch.int32, device=a.device)
    else:
        pivot = out_pivot
    # Allocate tensor to report info
    if out_info is None:
        info = torch.empty((1,), dtype=torch.int32, device=a.device)
    else:
        info = out_info
    # Allocates output spinor
    if out_sgn is None:
        sgn = torch.empty((triton.cdiv(num_tile_m, BLOCK_SIZE_SGN_DIV) * BLOCK_SIZE_SGN_ACCUM,),
                          dtype=a.dtype, device=a.device)
    else:
        sgn = out_sgn
    # initialize synchronization using shared memory (faster than atomic)
    sync = torch.zeros((1,), dtype=torch.int32, device=a.device)
    # enqueue kernel
    with torch.cuda.device(a.device.index):
        while True:
            future = _lu_partial_pivoting[(num_tile_m, num_tile_n)](a, pivot, m, n, 0, info, sgn,
                                                                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                                    BLOCK_SIZE_K=32,
                                                                    num_warps
