import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def sum_std_kernel(
    input_ptr, output_sum_ptr, output_std_ptr,
    input_stride, output_stride, reduction_size,
    correction, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    input_offset = pid * input_stride + offsets

    # Load input elements
    input_data = tl.load(input_ptr + input_offset, mask=offsets < reduction_size, other=0.0)
    input_data = input_data.to(tl.float32)

    # Compute sum
    sum_val = tl.sum(input_data, axis=0)

    # Compute mean
    mean_val = sum_val / reduction_size

    # Compute variance
    variance = tl.sum((input_data - mean_val) ** 2, axis=0) / (reduction_size - correction)

    # Store results
    if tl.program_id(1) == 0:
        tl.store(output_sum_ptr + pid * output_stride, sum_val)
    if tl.program_id(1) == 1:
        tl.store(output_std_ptr + pid * output_stride, tl.sqrt(variance))

@torch.inference_mode()
def sum_std(input: Tensor, dim=None, keepdim=False, dtype=None, correction=1, out=None) -> Tensor:
    """
    Computes the sum of elements in the input tensor along the specified dimension(s),
    followed by calculating the standard deviation of the summed values.

    Args:
        input (Tensor): The input tensor.
        dim (int or tuple of ints, optional): The dimension(s) to reduce. If None, all dimensions are reduced.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default is False.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. If specified, the input tensor is cast to dtype before the operation. Default: None.
        correction (int, optional): Difference between the sample size and sample degrees of freedom. Default is 1 (Bessel's correction).
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The output tensor with sum and standard deviation computed.
    """
    if dtype is not None:
        input = input.to(dtype)

    # Determine reduction dimensions
    if dim is None:
        dim = tuple(range(input.dim()))
    elif isinstance(dim, int):
        dim = (dim,)

    # Compute the size of the reduction
    reduction_size = 1
    for d in dim:
        reduction_size *= input.size(d)

    # Compute strides
    input_stride = input.stride(dim[0])

    # Prepare output tensors
    if out is None:
        out_sum = torch.empty_like(input)
        out_std = torch.empty_like(input)
    else:
        out_sum, out_std = out

    # Launch Triton kernel
    BLOCK_SIZE = triton.next_power_of_2(reduction_size)
    grid = (input.numel() // reduction_size, 2)  # Launch grid with two programs for sum and std

    sum_std_kernel[grid](
        input, out_sum, out_std,
        input_stride, out_sum.stride(dim[0]), reduction_size,
        correction, BLOCK_SIZE,
        num_warps=4, num_stages=2
    )

    if not keepdim:
        out_sum = out_sum.squeeze(dim)
        out_std = out_std.squeeze(dim)

    return out_sum, out_std
