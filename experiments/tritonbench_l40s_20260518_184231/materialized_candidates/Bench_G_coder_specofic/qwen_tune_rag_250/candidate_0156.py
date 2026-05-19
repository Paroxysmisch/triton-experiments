# The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel, num_stages = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        grid = (1, 1, 1)
        signature = "*fp32,*fp32,i32,i32,i32"
        _, kernel = tc.compile(softmax_kernel, signature, constants={"BLOCK_SIZE": BLOCK_SIZE, "num_stages": num_stages},
                                size_limit=SIZE_SMEM, warm_cache_only=True, grid=grid, triton_options={})
        num_regs = kernel.num_regs
        num_stages = 1
        occupancy = NUM_REGS // (num_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, 8)
        if occupancy == 0:
            num_warps = 1
            num_stages = 2
            occupancy = NUM_REGS // (num_regs * WARP_SIZE * num_warps)
            occupancy = min(occupancy, 8)
        if occupancy == 0:
            num_warps = 1
            num_stages = 4
            occupancy = NUM_REGS // (num_regs * WARP_SIZE * num_warps)
            occupancy = min(occupancy, 8)
        num_stages = 1
        if num_warps > 4:
            num_stages = 2
        if num_warps > 8:
            num_stages = 4
        if num_warps > 16:
            num_stages = 8
        grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]), 1, 1)
        kernel, _ = tc.compile(softmax_kernel, signature, constants={"BLOCK_SIZE": BLOCK_SIZE, "num_stages": num_stages},
                                size_limit=SIZE_SMEM, warm_cache_only=True, grid=grid, triton_options={})
        kernels[BLOCK_SIZE] = (kernel, num_stages)

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]), 1, 1)
    num_stages = 1
    kernel[grid](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
                 num_warps=num_warps)
    return y

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        if dim is None:
            dim = -1
        else:
            dim = dim % x.ndim
        if x.stride(dim) <= 0:
            raise ValueError("softmax does not support decreasing dimensionality, please use view instead")
        shape = list(x.shape)
        n_rows = shape.pop(dim)
        num_rows = 1
        for s in shape:
            num_rows *= s
        # Flatten input to 2D tensor, preserving the natural memory order of the input tensor
        order = list(range(x.ndim))
        order.pop(dim)
        order.insert(0, dim)
        x = x.transpose(order).contiguous()
        x = x.view(num_rows, n_rows)
        y = softmax(x)
        ctx.save_for_backward(x)
        ctx.dim = dim
        return y

def softmax(x, dim=None, dtype=None):
    if dtype is None:
        dtype = x.dtype
    return Softmax.apply(x, dim, dtype)
