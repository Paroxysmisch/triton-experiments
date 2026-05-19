import triton
import triton.language as tl
import torch


@triton.jit
def _exp_sum_kernel(
    input_ptr,  # pointer to input tensor data
    partial_sum_ptr,  # pointer to partial sums
    N,  # total number of elements
    BLOCK_SIZE: tl.constexpr  # block size for 1D processing
):
    # Program index
    pid = tl.program_id(0)
    # Offsets for each thread within this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds when N is not multiple of BLOCK_SIZE
    mask = offsets < N

    # Load the input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute exponent
    x_exp = tl.exp(x)
    # Sum within the block
    block_sum = tl.sum(x_exp, axis=0)
    # Write out the partial sum
    tl.store(partial_sum_ptr + pid, block_sum)


def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    """
    Applies the exponential function to each element in the input tensor and then
    computes the mean of these exponentiated values. Optionally reduces along
    a specified dimension, keeps dimensions, casts to a specific dtype, or
    writes the result into 'out'.

    Args:
        input (torch.Tensor): The input tensor.
        dim (int or tuple of int, optional): Dimension(s) to reduce. If None,
            reduces over all elements.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
        dtype (torch.dtype, optional): Desired data type of the output.
        out (torch.Tensor, optional): Output tensor to store the result.

    Returns:
        torch.Tensor: Tensor of the mean of exponentiated values.
    """
    # If reducing over all elements (dim=None), flatten the input and run 1D kernel
    if dim is None:
        flat_input = input.contiguous().view(-1)
        N = flat_input.numel()

        # Decide block size (tune or fix)
        BLOCK_SIZE = 1024
        # Round up so we launch enough blocks
        grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )

        # Allocate partial-sum buffer
        partial_sum = torch.empty(grid[0], device=flat_input.device, dtype=flat_input.dtype)

        # Launch kernel
        _exp_sum_kernel[grid](
            flat_input, partial_sum,
            N,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Sum across all partial sums
        total_sum = partial_sum.sum()
        # Compute the mean
        mean_val = total_sum / N

        out_tensor = mean_val if out is None else out.copy_(mean_val)

        # Convert to requested dtype if specified
        if dtype is not None:
            out_tensor = out_tensor.to(dtype)

        # If keepdim is requested and dim=None, mimic the behavior by adding dimensions of size 1
        if keepdim:
            # produce a tensor of shape [1, 1, ..., 1]
            for _ in range(input.dim()):
                out_tensor = out_tensor.unsqueeze(0)

        return out_tensor

    # For reduction along specified dim(s), exponentiate first and then use torch mean
    # to simplify demonstration. You could extend this to a full Triton-based
    # multi-dimensional reduction if desired.
    exp_input = torch.exp(input)
    out_tensor = exp_input.mean(dim=dim, keepdim=keepdim)
    if dtype is not None:
        out_tensor = out_tensor.to(dtype)
    if out is not None:
        out.copy_(out_tensor)
        out_tensor = out
    return out_tensor
