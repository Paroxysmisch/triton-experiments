import math
import triton
import triton.language as tl
import torch

# Triton kernel for approximate polygamma computation via partial sum of series:
# polygamma(n, x) = (-1)^(n+1) * n! * sum_{k=0 to ∞} [1 / (x + k)^(n+1)]
# We truncate the infinite series to a fixed number of terms PARTIAL_SUM for demonstration purposes.
@triton.jit
def _polygamma_kernel(
    input_ptr, out_ptr,
    n, M,
    in_stride, out_stride,
    BLOCK_SIZE: tl.constexpr,
    PARTIAL_SUM: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    # Load input values
    x = tl.load(input_ptr + offsets * in_stride, mask=mask, other=0.0)

    # Compute factorial of n on the fly (small n assumed for demonstration)
    factorial = 1
    i = 1
    while i <= n:
        factorial = factorial * i
        i += 1

    # Compute parity factor = (-1)^(n+1)
    sign = 1.0 if ((n + 1) % 2 == 0) else -1.0

    # Approximate partial sum
    s = tl.zeros_like(x)
    for k in range(PARTIAL_SUM):
        s = s + 1.0 / tl.pow(x + float(k), n + 1)

    # Combine factors
    result = sign * float(factorial) * s

    # Store result
    tl.store(out_ptr + offsets * out_stride, result, mask=mask)


def polygamma(n, input, *, out=None):
    """
    Computes the n-th derivative of the digamma function (polygamma) on the given input tensor.
    n (int): the order of the derivative (nonnegative integer).
    input (Tensor): the input tensor.
    out (Tensor, optional): the output tensor. If not provided, a new tensor is created.
    
    Returns:
        Tensor: The resulting polygamma values.
    """
    assert isinstance(n, int) and n >= 0, "polygamma is implemented for nonnegative integer orders."
    assert input.is_cuda, "input must be a CUDA tensor."

    if out is None:
        out = torch.empty_like(input)

    # Flatten input/output for kernel
    M = input.numel()
    input_contig = input.contiguous()
    out_contig = out.contiguous()
    in_ptr = input_contig.data_ptr()
    out_ptr = out_contig.data_ptr()

    # Grid/block setup
    BLOCK_SIZE = 128
    grid = (math.ceil(M / BLOCK_SIZE),)

    # Launch Triton kernel
    _polygamma_kernel[grid](
        in_ptr,
        out_ptr,
        n,
        M,
        input_contig.stride(0),
        out_contig.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        PARTIAL_SUM=100,  # arbitrary truncation for demonstration
        num_warps=4
    )

    # Reshape output if needed
    if out is not out_contig:
        out.copy_(out_contig.view_as(out))

    return out
