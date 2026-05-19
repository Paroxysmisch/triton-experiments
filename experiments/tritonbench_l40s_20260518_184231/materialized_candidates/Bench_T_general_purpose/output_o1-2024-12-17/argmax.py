import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_flatten_kernel(
    data_ptr,  # pointer to input data
    idx_ptr,   # pointer to output indices
    N,         # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data from memory
    values = tl.load(data_ptr + offsets, mask=mask, other=-float('inf'))
    # Initialize local best
    best_val = values
    best_idx = offsets

    # WARP REDUCTION: reduce within the block
    length = BLOCK_SIZE
    while length > 1:
        half = (length + 1) // 2
        cond = tl.arange(0, BLOCK_SIZE) < half
        lhs_val = tl.where(cond, best_val, -float('inf'))
        rhs_val = tl.where(cond, tl.roll(best_val, half), -float('inf'))
        lhs_idx = tl.where(cond, best_idx, -1)
        rhs_idx = tl.where(cond, tl.roll(best_idx, half), -1)

        use_rhs = rhs_val > lhs_val
        best_val = tl.where(use_rhs, rhs_val, lhs_val)
        best_idx = tl.where(use_rhs, rhs_idx, lhs_idx)
        length = half
    # Write result for this block
    if tl.thread_id(0) == 0:
        tl.store(idx_ptr + pid, best_idx[0], mask=True)


@triton.jit
def _argmax_dim_kernel(
    data_ptr,    # pointer to input
    out_idx_ptr, # pointer to output indices
    stride_in,   # stride in dimension being reduced
    stride_out,  # stride for the output idx in that dimension
    dim_size,    # size of the dimension to reduce
    outer_size,  # number of outer elements
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row (outer element)
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    # row index in outer dimensions
    row = pid
    # pointer offset for this row
    row_offset_in = row * stride_in * dim_size
    row_offset_out = row * stride_out

    best_val = tl.full([BLOCK_SIZE], -float('inf'), tl.float32)
    best_idx = tl.full([BLOCK_SIZE], 0, tl.int32)

    # Iterate over segments in dim_size
    # Each thread iterates across the dimension being reduced
    for d in range(0, dim_size):
        val = tl.load(data_ptr + row_offset_in + (d * stride_in) + offsets, mask=offsets < BLOCK_SIZE)
        use_d = val > best_val
        best_val = tl.where(use_d, val, best_val)
        best_idx = tl.where(use_d, d, best_idx)

    if tl.arange(0, BLOCK_SIZE)[0] == 0:
        # store best index for the first element in block
        tl.store(out_idx_ptr + row_offset_out, best_idx[0])

def argmax(input: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.Tensor:
    """
    argmax(input, dim, keepdim=False) -> LongTensor
    """
    if dim is None:
        # Flatten and do a 1D argmax
        flattened = input.view(-1)
        N = flattened.numel()
        # Prepare output index
        out_idx = torch.empty(( (1,) if keepdim else () ), dtype=torch.long, device=input.device)
        # Grid
        BLOCK_SIZE = 1024
        grid = lambda meta: ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
        # Launch kernel
        idx_buf = torch.empty(((N + BLOCK_SIZE - 1) // BLOCK_SIZE,), dtype=torch.long, device=input.device)
        _argmax_flatten_kernel[grid](
            flattened.data_ptr(),
            idx_buf.data_ptr(),
            N,
            BLOCK_SIZE=BLOCK_SIZE
        )
        # Final CPU reduction among partial results
        final_val = float('-inf')
        final_idx = 0
        partial = idx_buf.cpu()
        data_cpu = flattened.cpu()
        for i in range(partial.numel()):
            if data_cpu[partial[i]] > final_val:
                final_val = data_cpu[partial[i]]
                final_idx = partial[i].item()
        out_idx.fill_(final_idx)
        return out_idx
    else:
        # Argmax along a specific dimension
        dim_size = input.size(dim)
        outer_size = int(input.numel() // dim_size)
        # Output size
        out_shape = list(input.shape)
        if not keepdim:
            del out_shape[dim]
        else:
            out_shape[dim] = 1
        out_idx = torch.empty(out_shape, dtype=torch.long, device=input.device)

        # Adjust for dimension
        stride_in = input.stride(dim)
        stride_out = out_idx.stride(dim) if keepdim else 1
        BLOCK_SIZE = 1  # each program processes one outer row in its entirety
        grid = (outer_size,)

        _argmax_dim_kernel[grid](
            input.data_ptr(),
            out_idx.data_ptr(),
            stride_in,
            stride_out,
            dim_size,
            outer_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out_idx
