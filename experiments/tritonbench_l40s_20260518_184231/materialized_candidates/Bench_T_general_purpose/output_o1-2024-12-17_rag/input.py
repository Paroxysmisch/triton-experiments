import torch
import triton
import triton.language as tl

_mean_kernels_cache = {}


@triton.jit
def _mean_kernel(
    input_ptr,                  # input tensor pointer
    output_ptr,                 # output tensor pointer
    stride_in,                  # stride for input dimension
    n_outer,                    # number of rows/outer dimension
    n_reduce,                   # number of elements being reduced (columns if row-reduce, etc.)
    BLOCK_SIZE: tl.constexpr,   # number of threads per block
    num_stages: tl.constexpr    # software pipelining stages
):
    # program_id(0) varies across the outer dimension (n_outer)
    pid = tl.program_id(0)
    # each program processes a single outer index
    if pid >= n_outer:
        return

    # pointer offsets
    input_row_ptr = input_ptr + pid * stride_in
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_reduce

    # loop to accumulate partial sum across the dimension being reduced
    # we accumulate sums of "BLOCK_SIZE" chunks at a time
    # partial_sum will hold the sum for the current row / outer slice
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # for dimension sizes bigger than BLOCK_SIZE, we stride in steps of BLOCK_SIZE
    # to load and sum up all elements
    for i in range(0, n_reduce, BLOCK_SIZE):
        offs = i + col_offsets
        in_mask = offs < n_reduce
        vals = tl.load(input_row_ptr + offs, mask=in_mask, other=0.0)
        partial_sum += vals

    # sum up partial_sum within the warp
    block_sum = tl.sum(partial_sum, axis=0)

    # compute mean
    mean_val = block_sum / n_reduce

    # store the result (we store a single float at index pid)
    # output_ptr is shaped so that output_ptr[pid] is the correct location
    if pid < n_outer:
        tl.store(output_ptr + pid, mean_val)


def mean(input, dim, keepdim=False, dtype=None, out=None):
    """
    Returns the mean value of the input tensor across the specified dimension(s). 
    If dim is a list/tuple of dimensions, reduce over them one at a time.
    If keepdim is True, the reduced dimensions are retained with size 1.
    If dtype is specified, the input is cast to that dtype before the operation.
    If out is specified, the result is stored in it.
    """
    # handle optional casting
    if dtype is not None and input.dtype != dtype:
        input = input.to(dtype)

    # wrap a single int dim in a tuple
    if isinstance(dim, int):
        dim = (dim,)

    # reduce each dimension in dim one by one
    # sort dims to avoid dimension change issues after squeeze
    sorted_dims = sorted(dim)
    result = input

    for d in sorted_dims:
        # ensure d is positive
        d = d if d >= 0 else (d + result.ndim)
        # current dimension info
        size_reduce = result.shape[d]

        # if we keep dimension, shape after reduction has size 1 at dim d
        # otherwise, that dimension will be removed
        out_shape = list(result.shape)
        out_shape[d] = 1

        # create output buffer if not keepdim; else shape is with dimension = 1
        # we only do one dimension reduction at a time
        if keepdim:
            tmp_shape = out_shape
        else:
            tmp_shape = [result.shape[i] for i in range(result.ndim) if i != d]

        # flatten so that the dimension d to be reduced is the last dimension
        # or first dimension, to match row-based or column-based approach
        # here we put d as the last dimension for convenience
        perm = list(range(result.ndim))
        perm[d], perm[-1] = perm[-1], perm[d]
        transposed = result.permute(perm)
        n_outer = transposed.shape[0:-1]
        n_reduce = transposed.shape[-1]
        # flatten all outer dims into a single outer dimension
        outer_size = 1
        for s in n_outer:
            outer_size *= s

        # contiguous for direct pointer access
        transposed_contig = transposed.contiguous()
        inp_ptr = transposed_contig.data_ptr()

        # prepare shapes for the output of this single dim reduction
        if keepdim:
            # shape remains the same perm aside from dimension d which is 1
            out_trans_shape = list(transposed.shape)
            out_trans_shape[-1] = 1
        else:
            out_trans_shape = list(transposed.shape[:-1])

        # allocate intermediate output
        # it will hold the partial reduction result for this dimension
        tmp_out = torch.empty(out_trans_shape, dtype=transposed_contig.dtype, device=transposed_contig.device)
        out_ptr = tmp_out.data_ptr()

        # block size: nearest power of two for the reduce dimension
        BLOCK_SIZE = triton.next_power_of_2(n_reduce)
        # limit block size for large dims
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

        # craft the kernel
        # for demonstration, let num_stages be 2 or 4
        # for larger dims we might want more staging
        num_stages = 2 if n_reduce < 2048 else 4
        grid = lambda meta: (outer_size,)

        # possibly cache or compile
        key = (BLOCK_SIZE, num_stages)
        if key not in _mean_kernels_cache:
            _mean_kernels_cache[key] = _mean_kernel

        mean_kernel_fn = _mean_kernels_cache[key]

        # launch kernel
        mean_kernel_fn[grid](
            inp_ptr,
            out_ptr,
            transposed_contig.stride(-2) if transposed_contig.ndim > 1 else 0,
            outer_size,
            n_reduce,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
        )

        # reshape tmp_out back to the transposed shape
        reduced_trans = tmp_out
        # undo the permute
        inv_perm = list(range(result.ndim))
        for i, p in enumerate(perm):
            inv_perm[p] = i

        if keepdim:
            # restore shape with dimension = 1 at index d
            reduced_trans = reduced_trans.permute(inv_perm).reshape(out_shape)
        else:
            reduced_trans = reduced_trans.permute(inv_perm).reshape(tmp_shape)

        result = reduced_trans

    # if 'out' is specified, copy result to out
    if out is not None:
        out.copy_(result)
        return out
    return result
