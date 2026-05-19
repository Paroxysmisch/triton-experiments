import triton
import triton.language as tl
import torch

@triton.jit
def _fft1d_kernel(
    data_real_ptr, data_imag_ptr,
    n, stage,
    BLOCK_SIZE: tl.constexpr
):
    # stage-based Cooley-Tukey butterfly
    # n is length of 1D segment (must be power of 2)
    # stage indicates which butterfly size we are at: 2^stage
    pid = tl.program_id(axis=0)
    base_idx = pid * BLOCK_SIZE
    idx = base_idx + tl.arange(0, BLOCK_SIZE)

    # stride for the current stage
    step = 1 << stage
    half_step = step >> 1

    # distance between pair elements
    pair_dist = n >> stage

    # load data
    real_val = tl.load(data_real_ptr + idx, mask=idx < n, other=0.0)
    imag_val = tl.load(data_imag_ptr + idx, mask=idx < n, other=0.0)

    # identify pos within the step
    group_idx = idx // step
    pos_in_group = idx & (step - 1)

    # compute twiddle factor
    twiddle_id = (pos_in_group % half_step) * pair_dist
    angle = -2.0 * 3.141592653589793 * twiddle_id / float(n)
    c = tl.cos(angle)
    s = tl.sin(angle)

    # gather butterfly mate
    mate_offset = half_step if pos_in_group < half_step else -half_step
    mate_real = tl.load(data_real_ptr + idx + mate_offset, mask=(idx + mate_offset) < n, other=0.0)
    mate_imag = tl.load(data_imag_ptr + idx + mate_offset, mask=(idx + mate_offset) < n, other=0.0)

    # compute butterfly
    # for "upper" half: X[k], for "lower" half: X[k + step/2]
    if pos_in_group >= half_step:
        # multiply mate by twiddle
        t_r = c * mate_real - s * mate_imag
        t_i = s * mate_real + c * mate_imag
        # new value
        new_r = real_val - t_r
        new_i = imag_val - t_i
        real_val = new_r
        imag_val = new_i
    else:
        t_r = c * mate_real - s * mate_imag
        t_i = s * mate_real + c * mate_imag
        new_r = real_val + t_r
        new_i = imag_val + t_i
        real_val = new_r
        imag_val = new_i

    # store results
    tl.store(data_real_ptr + idx, real_val, mask=idx < n)
    tl.store(data_imag_ptr + idx, imag_val, mask=idx < n)


@triton.jit
def _bitreverse_kernel(
    data_real_ptr, data_imag_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    # Classic bit-reversal permutation
    pid = tl.program_id(axis=0)
    base_idx = pid * BLOCK_SIZE
    idx = base_idx + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    val_r = tl.load(data_real_ptr + idx, mask=mask, other=0.0)
    val_i = tl.load(data_imag_ptr + idx, mask=mask, other=0.0)

    # Reverse bits
    # Since n is a power of 2, compute log2(n):
    logn = 0
    temp = n
    while temp > 1:
        temp >>= 1
        logn += 1

    # perform bit reversal
    j = tl.zeros_like(idx)
    i = idx
    for _ in range(logn):
        j <<= 1
        j |= (i & 1)
        i >>= 1

    # store in reversed position
    tl.store(data_real_ptr + j, val_r, mask=(j < n) & mask)
    tl.store(data_imag_ptr + j, val_i, mask=(j < n) & mask)


def _apply_fft1d_triton(x_real, x_imag, n):
    # bit-reversal
    grid = lambda META: ( (n + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'], )
    _bitreverse_kernel[grid](
        x_real, x_imag,
        n,
        BLOCK_SIZE=1024
    )

    # do log2(n) stages
    logn = n.bit_length() - 1
    for stage in range(logn):
        _fft1d_kernel[grid](
            x_real, x_imag,
            n, stage,
            BLOCK_SIZE=1024
        )


def fftn(input, s=None, dim=None, norm=None, *, out=None):
    """
    Triton-based wrapper for computing the N-dimensional FFT.
    Only supports powers-of-two signal lengths in all transformed dims
    and torch.half/torch.chalf on GPUs with SM53+.
    """
    # Handle default params
    if dim is None and s is None:
        dim = tuple(range(input.ndim))
        s = [input.size(d) for d in dim]
    elif dim is None and s is not None:
        dim = tuple(range(input.ndim - len(s), input.ndim))
    elif s is None and dim is not None:
        s = [input.size(d) for d in dim]

    if out is not None:
        out_tensor = out
    else:
        out_tensor = torch.empty_like(input, dtype=torch.cfloat if input.is_complex() else torch.cfloat)

    # cast input to complex float16 or float32
    # if input is real, promote to complex
    if input.dtype in [torch.float16, torch.half]:
        data = input.to(torch.complex32)
    elif input.dtype in [torch.complex32, torch.chalf]:
        data = input
    else:
        data = input.to(torch.complex64)

    # Possibly pad or trim per dimension
    # We'll do it dimension by dimension
    working = data
    for i, d in enumerate(dim):
        size_target = s[i]
        if size_target == -1:
            continue
        sz = working.size(d)
        if size_target < sz:
            # trim
            slices = []
            for j in range(working.ndim):
                if j == d:
                    slices.append(slice(0, size_target))
                else:
                    slices.append(slice(None))
            working = working[tuple(slices)]
        elif size_target > sz:
            # pad
            pad_amount = [(0, 0) for _ in range(working.ndim)]
            pad_amount[d] = (0, size_target - sz)
            working = torch.nn.functional.pad(working, tuple(reversed([x for pair in pad_amount for x in pair])))

    # Ensure shape is correct
    shape_after = list(working.shape)
    # transform along dim in order
    for d in dim:
        n = shape_after[d]
        # check power-of-two
        if n & (n - 1) != 0:
            raise ValueError("All transformed dimensions must be powers of 2 for this Triton FFT.")
        # move d to last dimension by permuting
        perm = list(range(working.ndim))
        perm[d], perm[-1] = perm[-1], perm[d]
        working = working.permute(perm)
        # flatten all but last dimension
        leading_size = 1
        for sz_i in working.shape[:-1]:
            leading_size *= sz_i
        # reshape
        working = working.reshape(leading_size, n)
        # separate real and imaginary
        real_t = working.real.contiguous()
        imag_t = working.imag.contiguous()
        # convert to pinned memory on CUDA
        if working.is_cuda:
            real_t = real_t.clone()
            imag_t = imag_t.clone()

        # run 1D fft on each row
        for row_index in range(leading_size):
            row_real = real_t[row_index]
            row_imag = imag_t[row_index]
            _apply_fft1d_triton(row_real, row_imag, n)

        # combine
        working = real_t + 1j * imag_t
        # restore shape
        working = working.reshape(*shape_after[:-1], n)
        # perm back
        perm_back = list(range(working.ndim))
        perm_back[-1], perm_back[d] = perm_back[d], perm_back[-1]
        working = working.permute(perm_back)

    # Apply normalization if needed
    size_prod = 1
    for i in s:
        if i != -1:
            size_prod *= i
    if norm == "forward":
        working /= float(size_prod)
    elif norm == "ortho":
        import math
        factor = 1.0 / math.sqrt(size_prod)
        working *= factor

    out_tensor.copy_(working)
    return out_tensor
