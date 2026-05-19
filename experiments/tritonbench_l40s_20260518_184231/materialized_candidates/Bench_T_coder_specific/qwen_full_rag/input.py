import torch
import triton
import triton.language as tl

@triton.jit
def _reduce_dim_kernel(func, inp, out, N, reduction_indices, strides, rfactor, **meta):
    pid = tl.program_id(axis=0)
    # start index
    ind = pid * rfactor
    # the last index of the current program
    last_ind = ind + rfactor
    # the index of the last element processed by the program
    last_elem_ind = last_ind - 1

    while ind < N:
        # We handle striding manually in the loop because some compilers have bug when lowering
        # indexed assignment with complex stride expressions.
        row_start_ptr = inp + ind * strides[0]
        row_start_out_ptr = out + ind * strides[0]

        last_row_start_ptr = inp + last_elem_ind * strides[0]
        ld_mask = (ind < N) & (tl.arange(0, meta['BLOCK']) + ind <= last_elem_ind)
        rev_ld_mask = (ind < N) & (tl.arange(0, meta['BLOCK']) + ind > last_elem_ind)

        i = tl.arange(0, meta['BLOCK'])
        buf = tl.load(row_start_ptr + i, mask=ld_mask, other=float(0))
        buf = func(buf, axis=0)

        tl.store(row_start_out_ptr + i, buf, mask=ld_mask)
        if meta['REVERSE']:
            buf = tl.load(last_row_start_ptr - i, mask=rev_ld_mask, other=float(0))
            buf = func(buf, axis=0)
            tl.store(row_start_out_ptr - i, buf, mask=rev_ld_mask)

        ind += meta['BLOCK']

def reduce_dim(inp, func, dim=None, keepdim=False, dtype=None, out=None):
    """Reduces the input tensor along the given dimension using the specified function."""
    if dtype is None:
        dtype = inp.dtype
    if dtype != inp.dtype:
        inp = inp.to(dtype)

    if out is None:
        out = torch.empty_like(inp)
    else:
        assert out.shape == inp.shape

    if keepdim is False:
        out = out.squeeze(dim)

    inp = inp.contiguous()

    shape = list(inp.shape)
    if isinstance(dim, (list, tuple)):
        dim = sorted(dim)
        stride = inp.stride()
        reverse_dims = [False for _ in stride]
        for d in dim:
            assert (
                d >= -inp.dim() and d < inp.dim()
            ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -inp.dim(), inp.dim() - 1, d
            )
            assert not reverse_dims[
                d
            ], "dim {} appears multiple times in the list of dims".format(d)
            reverse_dims[d] = True
        reverse = any(reverse_dims)
        ndim = len(dim)
        reduced_shape = [
            (shape[i] + (shape[i] % meta['BLOCK'] - 1)) // meta['BLOCK']
            for i, meta in enumerate(zip(shape, stride))
            if not reverse_dims[i]
        ] + [(shape[i] + (shape[i] % meta['BLOCK'] - 1)) // meta['BLOCK']]
        for i in range(ndim - 1, -1, -1):
            if reverse_dims[i]:
                reduced_shape.insert(i, (shape[i] + (shape[i] % meta['BLOCK'] - 1)) // meta['BLOCK'])

        out = out.reshape(reduced_shape)
        M = 1
        for d in dim:
            M *= shape[d]
        BLOCK_M = triton.next_power_of_2(M)
        num_stages = 4 if META['SIZE_SMEM'] > 200000 else 2
        _reduce_dim_kernel[(M, )](
            func,
            inp,
            out,
            M,
            dim,
            inp.stride(),
            BLOCK_M // BLOCK,
            BLOCK=BLOCK,
            REVERSE=reverse,
            num_stages=num_stages,
        )
    else:
        assert dim >= -inp.dim() and dim < inp.dim(), "Invalid dim"
        dim = dim % inp.ndim
        N = shape[dim]
        shape[dim] = (N + BLOCK - 1) // BLOCK
        out = out.reshape(shape)
        if N % BLOCK == 0:
            num_programs = shape[dim]
        else:
            num_programs = shape[dim] - 1
        shape[dim] = N
        inp = inp.reshape(shape)
        _reduce_dim_kernel[(num_programs, )](func, inp, out, N, (dim,), inp.stride(), BLOCK, num_stages=2)
    return out
