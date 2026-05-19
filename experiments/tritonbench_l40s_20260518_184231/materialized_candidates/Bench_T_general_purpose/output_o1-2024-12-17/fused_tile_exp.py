import torch
import triton
import triton.language as tl


@triton.jit
def _tile_exp_kernel(
    input_ptr, output_ptr,
    in_d0, in_d1, in_d2, in_d3,
    out_d0, out_d1, out_d2, out_d3,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute multi-dimensional index for the output
    # (assuming up to 4D, extra dimensions can be set to 1)
    out_idx0 = offsets % out_d3
    tmp1 = offsets // out_d3
    out_idx1 = tmp1 % out_d2
    tmp2 = tmp1 // out_d2
    out_idx2 = tmp2 % out_d1
    out_idx3 = tmp2 // out_d1

    # Compute the corresponding input index by mod
    in_idx3 = out_idx3 % in_d0
    in_idx2 = out_idx2 % in_d1
    in_idx1 = out_idx1 % in_d2
    in_idx0 = out_idx0 % in_d3

    # Convert to linear indices for both input and output
    in_lin_idx = (
        in_idx3 * (in_d1 * in_d2 * in_d3)
        + in_idx2 * (in_d2 * in_d3)
        + in_idx1 * in_d3
        + in_idx0
    )
    out_lin_idx = (
        out_idx3 * (out_d1 * out_d2 * out_d3)
        + out_idx2 * (out_d2 * out_d3)
        + out_idx1 * out_d3
        + out_idx0
    )

    # Load input, compute exp, store to output
    x = tl.load(input_ptr + in_lin_idx, mask=mask, other=0.0)
    x_exp = tl.exp(x)
    tl.store(output_ptr + out_lin_idx, x_exp, mask=mask)


def fused_tile_exp(input, dims, *, out=None):
    """
    fused_tile_exp(input, dims, *, out=None) -> Tensor

    Performs a fused operation combining tiling and the exponential function:
    1) Tiles `input` according to the specified `dims`.
    2) Applies exp element-wise to the tiled result.

    Arguments:
        input (Tensor): The input tensor whose elements are to be repeated and exponentiated.
        dims (tuple of int): The number of repetitions for each dimension.
                            If shorter than input.dim(), 1's are prepended.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.
    """
    # Ensure dims covers all input dimensions by prepending ones if necessary
    if len(dims) < input.dim():
        dims = (1,) * (input.dim() - len(dims)) + tuple(dims)

    in_shape = input.shape
    # Compute output shape by multiplying each dimension in in_shape with dims
    out_shape = [in_shape[i] * dims[i] for i in range(input.dim())]

    # Expand to 4D if needed (pad with 1s at the front)
    max_ndim = 4
    padded_in = (1,) * (max_ndim - len(in_shape)) + in_shape
    padded_out = (1,) * (max_ndim - len(out_shape)) + tuple(out_shape)

    # Prepare output tensor
    if out is None:
        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    # Flatten all shapes for passing to the kernel
    in_d0, in_d1, in_d2, in_d3 = padded_in
    out_d0, out_d1, out_d2, out_d3 = padded_out

    n_elements = out.numel()
    # Launch the kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _tile_exp_kernel[grid](
        input, out,
        in_d0, in_d1, in_d2, in_d3,
        out_d0, out_d1, out_d2, out_d3,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
