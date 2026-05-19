import torch
import triton
import triton.language as tl

# Utility function to check if a given size is a power of two
def _is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n > 0

@triton.jit
def _fft1d_kernel(
    real_ptr, imag_ptr,
    real_out_ptr, imag_out_ptr,
    stride_src, stride_dst,
    length, # must be a power of 2
    BLOCK_SIZE: tl.constexpr
):
    """
    1D Cooley-Tukey FFT kernel for a power-of-two length.
    Operates on a single batch at a time.
    """
    pid = tl.program_id(0)
    # offset in input and output
    real_src = real_ptr + pid * stride_src
    imag_src = imag_ptr + pid * stride_src
    real_dst = real_out_ptr + pid * stride_dst
    imag_dst = imag_out_ptr + pid * stride_dst

    # We'll map each thread to an index in the 1D array
    idx = tl.arange(0, BLOCK_SIZE)
    # If length < BLOCK_SIZE, some threads are inactive
    mask = idx < length

    # Load from input
    # For simplicity, assume length == BLOCK_SIZE in this example
    real_val = tl.load(real_src + idx, mask=mask, other=0.0)
    imag_val = tl.load(imag_src + idx, mask=mask, other=0.0)

    # Iterative Cooley-Tukey
    half = 1
    while half < length:
        step = half << 1
        angle = -3.141592653589793 * 2.0 / step  # negative for forward transform
        twiddle_idx = idx & (half - 1)
        twiddle_factor = twiddle_idx * angle

        # Butterfly
        cos_val = tl.cos(twiddle_factor)
        sin_val = tl.sin(twiddle_factor)

        partner = idx ^ half
        real_partner = tl.load(real_src + partner, mask=mask, other=0.0)
        imag_partner = tl.load(imag_src + partner, mask=mask, other=0.0)

        # Multiply by twiddle
        temp_r = real_partner * cos_val - imag_partner * sin_val
        temp_i = real_partner * sin_val + imag_partner * cos_val

        # Write top or bottom half
        if_else = (idx & half) != 0
        # top half
        real_temp = real_val - temp_r
        imag_temp = imag_val - temp_i
        # bottom half
        real_val = tl.where(if_else, real_temp, real_val + temp_r)
        imag_val = tl.where(if_else, imag_temp, imag_val + temp_i)

        # Barrier so changes are visible for next iteration
        tl.store(real_src + idx, real_val, mask=mask)
        tl.store(imag_src + idx, imag_val, mask=mask)
        tl.barrier()

        half = step

    # Store output
    tl.store(real_dst + idx, real_val, mask=mask)
    tl.store(imag_dst + idx, imag_val, mask=mask)


def _fft1d(real_in, imag_in, len_dim):
    """
    Python helper to call the 1D Triton kernel for each batch.
    real_in/imag_in are 1D or collapsed views for the dimension to transform.
    len_dim is the size of the transform dimension (power of two).
    """
    # We expect real_in, imag_in to have shape [batch, len_dim]
    batch = real_in.shape[0]
    out_real = torch.empty_like(real_in)
    out_imag = torch.empty_like(imag_in)

    # Launch kernel for each batch
    grid = (batch,)
    BLOCK_SIZE = len_dim
    triton.run(
        _fft1d_kernel,
        grid=grid,
        num_warps=1,
        num_stages=2,
        args=[
            real_in, imag_in,
            out_real, out_imag,
            len_dim, len_dim,
            len_dim,
        ],
        constants={"BLOCK_SIZE": BLOCK_SIZE}
    )
    return out_real, out_imag


def fftn(input, s=None, dim=None, norm=None, *, out=None):
    """
    fftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

    Parameters:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
        dim (Tuple[int], optional): Dimensions to be transformed.
        norm (str, optional): Normalization mode. One of:
            'forward'  - normalize by 1/n
            'backward' - no normalization
            'ortho'    - normalize by 1/sqrt(n)
        out (Tensor, optional): the output tensor
    """
    # Handle dim
    if dim is None:
        if s is None:
            dim = tuple(range(input.dim()))
        else:
            dim = tuple(range(input.dim() - len(s), input.dim()))
    # Default s
    if s is None:
        s = [input.size(d) for d in dim]
    else:
        s = list(s)

    # Prepare output
    if out is not None:
        out_tensor = out
    else:
        out_tensor = torch.empty_like(input, dtype=torch.cfloat if input.is_complex() else torch.cfloat)

    x = input.to(torch.cfloat)
    # Possibly pad or trim
    for i, d in enumerate(dim):
        size_needed = s[i]
        if size_needed != -1 and size_needed != x.size(d):
            # zero-pad or trim
            sizes = list(x.shape)
            if size_needed < x.size(d):
                # trim
                slice_obj = [slice(None)] * x.dim()
                slice_obj[d] = slice(0, size_needed)
                x = x[tuple(slice_obj)].clone()
            else:
                # pad
                pad_before = [0, ] * x.dim() * 2
                pad_before[d * 2 + 1] = size_needed - x.size(d)
                x = torch.nn.functional.pad(x, pad_before)

    # Recompute dim in case shape changed
    # (some dimensions might have changed with padding/trim)
    if dim is None:
        dim = tuple(range(x.dim()))
    # Then do repeated 1D transforms for each dimension
    curr = x
    for d, size_d in zip(dim, s):
        if size_d != -1:
            # We must have size_d as a power of two
            if not _is_power_of_two(size_d):
                raise ValueError("Triton FFT kernel only supports power-of-two lengths.")
            # Move the dimension d to the last dimension
            perm = list(range(curr.dim()))
            perm[d], perm[-1] = perm[-1], perm[d]
            curr = curr.permute(perm)
            shape_c = curr.shape
            batch = 1
            for bsz in shape_c[:-1]:
                batch *= bsz

            reals = curr.real.contiguous().view(batch, shape_c[-1])
            imags = curr.imag.contiguous().view(batch, shape_c[-1])
            out_real, out_imag = _fft1d(reals, imags, shape_c[-1])
            fft_result = out_real + 1j * out_imag
            fft_result = fft_result.view(*shape_c)
            # Move dimension back
            fft_result = fft_result.permute(perm)
            curr = fft_result

    # Apply normalization if needed
    nprod = 1
    for size_d in s:
        if size_d != -1:
            nprod *= size_d
    if norm == 'forward':
        curr = curr / nprod
    elif norm == 'ortho':
        curr = curr / (nprod ** 0.5)

    out_tensor.copy_(curr)
    return out_tensor
