import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

# Define the Triton kernel for computing sum and standard deviation
@triton.jit
def sum_std_kernel(
    input,
    output_sum,
    output_std,
    input_row_stride,
    n_elements,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)

    # Load input slice
    x_ptr = input + block_start
    x = tl.load(x_ptr, mask=(block_end > block_start))

    # Compute local sums and squared sums
    local_sum = tl.sum(x, axis=0)
    local_sq_sum = tl.sum(x * x, axis=0)

    # Reduce across blocks
    sum_shared = tl.zeros((BLOCK_SIZE,), dtype=x.dtype)
    sq_sum_shared = tl.zeros((BLOCK_SIZE,), dtype=x.dtype)
    tl.reduce(sum_shared, local_sum, op=tl.math.fadd, axis=0)
    tl.reduce(sq_sum_shared, local_sq_sum, op=tl.math.fadd, axis=0)

    # Allreduce across warps
    if BLOCK_SIZE >= 512:
        tl.sync_mem()
    if BLOCK_SIZE >= 256:
        tl.reduce(sum_shared, sum_shared[0:256], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:256], op=tl.math.fadd, axis=0)
    if BLOCK_SIZE >= 128:
        tl.sync_mem()
        tl.reduce(sum_shared, sum_shared[0:128], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:128], op=tl.math.fadd, axis=0)
    if BLOCK_SIZE >= 64:
        tl.sync_mem()
        tl.reduce(sum_shared, sum_shared[0:64], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:64], op=tl.math.fadd, axis=0)
    if BLOCK_SIZE >= 32:
        tl.sync_mem()
        tl.reduce(sum_shared, sum_shared[0:32], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:32], op=tl.math.fadd, axis=0)
    if BLOCK_SIZE >= 16:
        tl.sync_mem()
        tl.reduce(sum_shared, sum_shared[0:16], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:16], op=tl.math.fadd, axis=0)
    if BLOCK_SIZE >= 8:
        tl.sync_mem()
        tl.reduce(sum_shared, sum_shared[0:8], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:8], op=tl.math.fadd, axis=0)
    if BLOCK_SIZE >= 4:
        tl.sync_mem()
        tl.reduce(sum_shared, sum_shared[0:4], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:4], op=tl.math.fadd, axis=0)
    if BLOCK_SIZE >= 2:
        tl.sync_mem()
        tl.reduce(sum_shared, sum_shared[0:2], op=tl.math.fadd, axis=0)
        tl.reduce(sq_sum_shared, sq_sum_shared[0:2], op=tl.math.fadd, axis=0)

    # Write results back to global memory
    if pid == 0:
        total_sum = sum_shared[0]
        total_sq_sum = sq_sum_shared[0]
        mean = total_sum / n_elements
        variance = total_sq_sum / n_elements - mean * mean
        std_dev = tl.sqrt(variance / max(0, n_elements - correction))
        tl.store(output_sum, total_sum)
        tl.store(output_std, std_dev)

# Wrapper function to compute sum and standard deviation
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
        Tensor: The output tensor containing the sum and standard deviation.
    """

    if dim is None:
        dim = list(range(input.dim()))
    elif isinstance(dim, int):
        dim = [dim]

    # Flatten the input tensor to perform reduction
    flat_input = input.flatten()

    # Allocate output tensors
    output_sum = torch.zeros(1, dtype=input.dtype, device=input.device)
    output_std = torch.zeros(1, dtype=input.dtype, device=input.device)

    # Determine the block size
    BLOCK_SIZE = 1024

    # Launch the kernel
    grid_size = (flat_input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    sum_std_kernel[grid_size](flat_input, output_sum, output_std, flat_input.stride(0), flat_input.numel(), correction, BLOCK_SIZE, num_warps=4, num_stages=2)

    # Unflatten the output and reshape if necessary
    result = torch.cat([output_sum, output_std]).view(2)
    if out is not None:
        result.copy_(out)

    return result
