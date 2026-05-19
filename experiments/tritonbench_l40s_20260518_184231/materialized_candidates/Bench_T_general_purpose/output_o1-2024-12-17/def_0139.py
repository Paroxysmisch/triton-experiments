import triton
import triton.language as tl
import torch

@triton.jit
def _std_sum_kernel(
    x_ptr,  # input data
    sum_ptr,  # partial sums
    sq_sum_ptr,  # partial squred sums
    N,  # number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x

    # compute partial sums
    partial_sum = tl.sum(x, axis=0)
    partial_sq_sum = tl.sum(x_sq, axis=0)

    # write out
    if tl.thread_id(0) == 0:
        tl.store(sum_ptr + pid, partial_sum)
        tl.store(sq_sum_ptr + pid, partial_sq_sum)

@triton.jit
def _std_finalize_kernel(
    sum_ptr,
    sq_sum_ptr,
    out_ptr,
    total_elements,
    correction,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # load partial sums
    partial_sum = tl.load(sum_ptr + pid)
    partial_sq_sum = tl.load(sq_sum_ptr + pid)

    # store them temporarily in out buffer
    tl.store(out_ptr + pid * 2 + 0, partial_sum)
    tl.store(out_ptr + pid * 2 + 1, partial_sq_sum)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    # parse and handle dim
    if dim is None:
        # reduce all dims
        dims_to_reduce = list(range(input.ndim))
    elif isinstance(dim, int):
        dims_to_reduce = [dim]
    else:
        dims_to_reduce = list(dim)

    # move input to contiguous for reduction
    x = input
    for d in sorted(dims_to_reduce, reverse=True):
        x = x.transpose(d, x.ndim-1)
    reduce_size = 1
    for d in dims_to_reduce:
        reduce_size *= input.shape[d]
    other_dims = [s for i, s in enumerate(input.shape) if i not in dims_to_reduce]
    flattened = x.reshape(-1, reduce_size).contiguous()

    # allocate intermediate results on device
    device = flattened.device
    num_segments = flattened.shape[0]  # number of rows after flatten
    block_size = 1024
    grid = ( (reduce_size + block_size - 1) // block_size, )

    sum_buf = torch.empty(num_segments, dtype=flattened.dtype, device=device)
    sq_sum_buf = torch.empty(num_segments, dtype=flattened.dtype, device=device)

    # launch kernel for partial sums
    _std_sum_kernel[grid](
        flattened.data_ptr(),
        sum_buf.data_ptr(),
        sq_sum_buf.data_ptr(),
        reduce_size,
        BLOCK_SIZE=block_size
    )

    # finalize partial sums
    final_buf = torch.empty(num_segments * 2, dtype=flattened.dtype, device=device)
    _std_finalize_kernel[ (num_segments,) ](
        sum_buf.data_ptr(),
        sq_sum_buf.data_ptr(),
        final_buf.data_ptr(),
        reduce_size,
        correction,
        BLOCK_SIZE=1
    )

    # move results to cpu for final calc
    final_vals = final_buf.cpu().view(num_segments, 2)
    sums = final_vals[:, 0]
    sq_sums = final_vals[:, 1]

    # compute global sum per segment
    mean_vals = sums / reduce_size
    var_vals = (sq_sums / reduce_size) - mean_vals**2
    # apply correction
    denom = max(1, reduce_size - correction)
    var_vals_corrected = var_vals * (reduce_size / denom)

    std_vals = torch.sqrt(var_vals_corrected)

    # reshape to correct output shape
    out_shape = other_dims if not keepdim else [
        (1 if i in dims_to_reduce else s) for i, s in enumerate(input.shape)
    ]
    if not keepdim:
        # after removing dims_to_reduce, we form the shape from 'other_dims'
        pass

    result = std_vals.view(*out_shape)

    if out is not None:
        out.copy_(result)
        return out
    return result
