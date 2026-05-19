_jl/dx_lk * dy_ij)
    -            = sum_i(dy_il * dy_il) + sum_j(dy_jl * dy_ij)

    This function works from the output perspective.
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


def pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Adaptive average pooling
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size)

    # Compute pairwise distance
    dist = torch.cdist(x1_pooled, x2_pooled, p=p, eps=eps)

    if keepdim:
        return dist
    else:
        return dist.squeeze()


def pairwise_distance_adaptive_avg_pool2d_grad(grad_output: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Adaptive average pooling
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size)

    # Compute pairwise distance
    dist = torch.cdist(x1_pooled, x2_pooled, p=p, eps=eps)

    # Compute gradients
    grad_x1 = torch.autograd.grad(dist, x1, grad_output, retain_graph=True)[0]
    grad_x2 = torch.autograd.grad(dist, x2, grad_output, retain_graph=True)[0]

    if keepdim:
        return grad_x1, grad_x2
    else:
        return grad_x1.squeeze(), grad_x2.squeeze()


def pairwise_distance_adaptive_avg_pool2d_triton(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Adaptive average pooling
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size)

    # Compute pairwise distance
    dist = torch.cdist(x1_pooled, x2_pooled, p=p, eps=eps)

    if keepdim:
        return dist
    else:
        return dist.squeeze()


def pairwise_distance_adaptive_avg_pool2d_triton_grad(grad_output: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Adaptive average pooling
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size)

    # Compute pairwise distance
    dist = torch.cdist(x1_pooled, x2_pooled, p=p, eps=eps)

    # Compute gradients
    grad_x1 = torch.autograd.grad(dist, x1, grad_output, retain_graph=True)[0]
    grad_x2 = torch.autograd.grad(dist, x2, grad_output, retain_graph=True)[0]

    if keepdim:
        return grad_x1, grad_x2
    else:
        return grad_x1.squeeze(), grad_x2.squeeze()


def pairwise_distance_adaptive_avg_pool2d_triton_grad_triton(grad_output: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0, eps: float = 1e-6, keep
