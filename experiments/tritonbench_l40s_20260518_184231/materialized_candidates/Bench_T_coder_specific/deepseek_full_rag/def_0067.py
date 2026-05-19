dy_ij/dx_lk * dy_ij)
    - x_lk_grad = -2 * sum_i(x_im * dy_il) - 2 * sum_j(x_jm * dy_ij) if i != j
    - x_lk_grad = 2 * sum_i(x_im * dy_il) if i == j
    - x_lk_grad = 0 otherwise

    This function works from the output and input gradient perspective.
    The goal is to compute every pairwise distance between each row of x_ptr
    and store this value in dist_sq_ptr for all the NxN values

    xnumel = N*N
    rnumel = D
    """
    # Compute output values offset
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.expand_dims(xoffset + tl.arange(0, XBLOCK), 1)  # XBLOCK, 1

    # We guard against going out of the output matrix number of elements
    # This guard will also be used to avoid loading already computed output values
    xmask = xindex < xnumel

    # First trick:
    # This is nice way map output values to input indexes and compute all the pairwise distances
    # While ensuring we never go beyond the number of lines of the input matrix
    pw_row1_index = D * (xindex // N)  # [0, ..., 0, 1, ...]
    pw_row2_index = D * (xindex % N)  # [0, 1, ..., N-1, 0, ...]

    # We allocate the memory to store the temporary values
    acc_add = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    # Dynamic reduction working per block
    rbase = tl.expand_dims(tl.arange(0, RBLOCK), 0)  # 1, RBLOCK
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel  # We guard against going out of the first dimension

        # Second trick
        # We use an outer AND and outer + operations to get the current reduction indexes for
        # the rows handled by the current block
        mask = rmask & xmask  # XBLOCK, RBLOCK
        in1_ptrs = pw_row1_index + rindex  # XBLOCK, RBLOCK
        in0_ptrs = pw_row2_index + rindex  # XBLOCK, RBLOCK

        data0 = tl.load(x_ptr + in1_ptrs, mask, eviction_policy="evict_last", other=0)
        data1 = tl.load(x_ptr + in0_ptrs, mask, eviction_policy="evict_last", other=0)

        # We do all the pointwise operations
        diff = data0 - data1
        diff_squared = diff * diff

        # This line is not needed because we are sure that we are working with
        # tensors of size [XBLOCK, RBLOCK] already
        # diff_squared_brodcasted = tl.broadcast_to(diff_squared, [XBLOCK, RBLOCK])

        # Those lines can be simplified because we mask our input values with the 0. value
        # and (0 - 0)**2 -> 0 so it won't interfere with the accumulation
        acc_add += diff_squared
        # # We add to our temporary buffer
        # # and make sure to only keep the values that has been updated
        # tmp6 = acc_add + diff_squared
        # acc_add = tl.where(mask, tmp6, acc_add)

    # We finally reduce to get final output values
    row_sum = tl.expand_dims(tl.sum(acc_add, 1), 1)
    # And we write back in the global memory
    tl.store(dist_sq_ptr + xindex, row_sum, xmask)


class PairwiseDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p=2.0, eps=1e-6, keepdim=False):
        N, D = x.shape
        xnumel = N * N
        rnumel = D

        # We create an output tensor to hold the result
        # We use contiguous because we want to make sure we have a contiguous memory
        # space for the output tensor
        dist_sq = torch.empty((N, N), device=x.device, dtype=torch.float32)

        # We call our kernel
        # We use N as the grid size because we want to run our kernel per line of the input
        _kole_dist_sq_forward[(xnumel,)](
            x, dist_sq, N, D, xnumel, rnumel, XBLOCK=N, RBLOCK=D
        )

        # We take the p-th root of the result
        # We use eps because we want to avoid division by zero
        dist = torch.pow(dist_sq + eps, 1.0 / p)

        ctx.save_for_backward(x)
        ctx.p = p
        ctx.eps = eps
        ctx.keepdim = keepdim

        return dist

    @staticmethod
    def backward(ctx, grad_dist):
        x, = ctx.saved_tensors
        p, eps, keepdim = ctx.p, ctx.eps, ctx.keepdim
        N, D = x.shape
        xnumel = N * N
        rnumel = D

        # We create an output tensor to hold the result
        # We use contiguous because we want to make sure we have a contiguous memory
        # space for the output tensor
        grad_x = torch.empty_like(x)

        # We call our kernel
        # We use N as the grid size because we want to run our kernel per line of the input
        _kole_dist_sq_backward[(xnumel,)](
            grad_dist, grad_x, x, N, D, xnumel, rnumel, XBLOCK=N, RBLOCK=D
        )

        return grad_x, None, None, None, None


def pairwise_distance(x, p=2.0, eps=1e-6, keepdim=False):
    return PairwiseDistance.apply(x, p, eps, keepdim)
