= dy_ik/dx_lk * dy_il + dy_kj/dx_lk * dy_lj for i != k, j != k
    - x_lk_grad = dy_kk/dx_lk * dy_lk + sum_j(dy_kj/dx_lk * dy_lj) if i = k
    - x_lk_grad = dy_ik/dx_lk * dy_il + dy_kj/dx_lk * dy_lj for j != k, i != k
    - x_lk_grad = -2 * dy_kk/dx_lk * dy_lk + sum_j(dy_kj/dx_lk * dy_lj) if i = k, j = k
    - x_lk_grad =  ....................................... sum_i(dy_il/dx_lk * dy_il)
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
    acc_add = tl.zeros([XBLOCK, RBLOCK], tl.float32)

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

        grad_dist_sq = tl.load(grad_dist_sq_ptr + xindex, mask, eviction_policy="evict_last", other=0)

        # We do all the pointwise operations
        acc_add += grad_dist_sq * diff

    # We finally reduce to get final output values
    row_sum = tl.expand_dims(tl.sum(acc_add, 1), 1)
    # And we write back in the global memory
    tl.store(grad_x_ptr + xindex, row_sum, xmask)


def normalize_pairwise_distance(
    x1: Tensor,
    x2: Tensor,
    p_distance: float = 2.0,
    eps_distance: float = 1e-6,
    keepdim: bool = False,
    p_norm: float = 2,
    dim_norm: int = 1,
    eps_norm: float = 1e-12,
) -> Tensor:
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    assert x1.is_contiguous(), "x1 must be a contiguous tensor"
    assert x2.is_contiguous(), "x2 must be a contiguous tensor"

    N, D = x1.shape
    xnumel = N * N
    rnumel = D

    # We compute all the pairwise distances
    dist_sq = torch.empty((N, N), device=x1.device, dtype=torch.float32)

    # We compute all the pairwise distances
    _kole_dist_sq_forward[(xnumel,)](
        x1,
        dist_sq,
        N,
        D,
        xnumel,
        rnumel,
        XBLOCK=triton.next_power_of_2(N),
        RBLOCK=triton.next_power_of_2(D),
    )

    # We take the square root to get the Euclidean distances
    dist = torch.sqrt(dist_sq + eps_distance)

    # We normalize
    dist = dist / (torch.max(dist, dim=dim_norm, keepdim=True).values + eps_distance)

    # We take the p-th power to obtain the desired norm
    dist = torch.pow(dist, p_distance)

    # We normalize x1 using its own distances
    x1_norm = torch.pow(torch.norm(x1.view(N, D) if keepdim else x1, p=p_norm, dim=dim_norm, keepdim=True) + eps_norm, p_norm / p_distance)

    # We combine the two operations
    normalized_distance = dist * x1_norm

    return normalized_distance
