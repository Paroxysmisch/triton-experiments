import triton
import triton.language as tl
import torch

@triton.jit
def _logsumexp_kernel(
    input_ptr, output_ptr,
    n_elements, row_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes a single row
    row_id = tl.program_id(0)
    row_offset = row_id * row_stride

    # Partial maximum for stability
    max_val = tl.float32(-1e30)
    # First pass: find row-wise max
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=-1e30)
        max_val = tl.maximum(max_val, tl.maximum(x, -1e30))

    # Partial sum of exponentials
    sum_exp = tl.float32(0)
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=-1e30)
        sum_exp += tl.sum(tl.exp(x - max_val), where=mask)

    # Final log-sum-exp
    lse = max_val + tl.log(sum_exp)
    # Store the result
    tl.store(output_ptr + row_id, lse)


def logsumexp(input, dim, keepdim=False, *, out=None):
    # Convert dim if negative
    dim = (dim + input.ndim) % input.ndim
    # Permute so that 'dim' is the last dimension
    perm_order = list(range(input.ndim))
    perm_order[-1], perm_order[dim] = perm_order[dim], perm_order[-1]
    x = input.permute(perm_order)
    shape = x.shape
    rows, cols = shape[:-1], shape[-1]

    # Flatten leading dims
    leading_size = 1
    for s in rows:
        leading_size *= s
    x_contig = x.contiguous().view(leading_size, cols)

    # Allocate output
    if out is None:
        out_shape = list(rows)
        if keepdim:
            out_shape.insert(dim, 1)
        result = torch.empty((leading_size,), dtype=x.dtype, device=x.device)
    else:
        # Assume 'out' has correct shape
        result = out.view(-1)

    # Launch Triton kernel
    grid = (leading_size,)
    block_size = 1024
    triton.run(
        _logsumexp_kernel,
        grid=grid,
        num_warps=4,
        BLOCK_SIZE=block_size,
        input_ptr=x_contig.data_ptr(),
        output_ptr=result.data_ptr(),
        n_elements=cols,
        row_stride=cols
    )

    # Reshape result
    result = result.view(*rows)
    # Permute back if needed
    if keepdim:
        result = result.unsqueeze(dim)
    if out is not None:
        out.copy_(result)
        return out
    return result
