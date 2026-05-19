import triton
import triton.language as tl
import torch

@triton.jit
def _softmax_log_kernel(
    input_ptr, 
    output_ptr, 
    N_COLS, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * N_COLS
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS

    # Load
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)

    # Apply log
    x_log = tl.log(x)

    # Exponentiate
    x_exp = tl.exp(x_log)

    # Compute sum
    s = tl.sum(x_exp, where=mask)

    # Final output
    out = x_exp / s

    # Store
    tl.store(output_ptr + row_start + offsets, out, mask=mask)


def softmax_log(input, dim=-1, dtype=None) -> torch.Tensor:
    """
    Applies the natural logarithm element-wise on the input tensor, 
    followed by applying the softmax function along the specified dimension.
    This combined operation scales input values to a range between 0 and 1, 
    summing to 1 after the logarithmic transformation.

    Args:
        input (Tensor): The input tensor on which logarithm and softmax are applied.
        dim (int): The dimension along which softmax will be computed. Default: -1.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. 
            If specified, the input tensor is cast to :attr:`dtype` before the operation is performed. 
            Useful for preventing data type overflows. Default: None.

    Example::
        >>> import torch
        >>> input = torch.rand(3, 4) * 10
        >>> result = softmax_log(input, dim=1)
        >>> result
        tensor([[0.1829, 0.1782, 0.2783, 0.3606],
                [0.3119, 0.1724, 0.3256, 0.1900],
                [0.2057, 0.2166, 0.2991, 0.2786]])

        >>> result = softmax_log(input, dim=0)
        >>> result
        tensor([[0.3122, 0.4444, 0.2720, 0.2159],
                [0.3879, 0.2167, 0.4226, 0.2165],
                [0.2999, 0.3389, 0.3055, 0.5676]])

    Math:
        out = Softmax(log(input))

        y_i = x_i / sum_j x_j
    """
    if dtype is not None:
        input = input.to(dtype)

    # Move requested dim to the last dimension
    dim = dim if dim >= 0 else input.ndim + dim
    perm = list(range(input.ndim))
    perm[-1], perm[dim] = perm[dim], perm[-1]
    input_ = input.permute(perm)

    # Flatten
    shape_ = input_.shape
    nrows = 1
    for s in shape_[:-1]:
        nrows *= s
    ncols = shape_[-1]

    # Allocate output
    input_contig = input_.contiguous().view(nrows, ncols)
    output_contig = torch.empty_like(input_contig)

    # Define block size (for simplicity, ensure ncols <= BLOCK_SIZE)
    BLOCK_SIZE = min(triton.next_power_of_2(ncols), 1024)
