import triton
import triton.language as tl

@triton.jit
def std_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    size,  # Total number of elements in the input tensor
    stride,  # Stride of the input tensor
    dim,  # Dimension(s) to reduce over
    correction,  # Bessel's correction
    keepdim,  # Whether to keep reduced dimensions
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the mean
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(block_start, block_start + BLOCK_SIZE):
        if i < size:
            mean += tl.load(input_ptr + i * stride)
    mean = mean / size

    # Compute the variance
    variance = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(block_start, block_start + BLOCK_SIZE):
        if i < size:
            x = tl.load(input_ptr + i * stride)
            variance += (x - mean) ** 2
    variance = variance / (size - correction)

    # Compute the standard deviation
    std = tl.sqrt(variance)

    # Write the result to the output tensor
    if keepdim:
        tl.store(output_ptr + block_start, std)
    else:
        tl.store(output_ptr + pid, std)

import torch
import triton
import triton.language as tl

@triton.jit
def std_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    size,  # Total number of elements in the input tensor
    stride,  # Stride of the input tensor
    dim,  # Dimension(s) to reduce over
    correction,  # Bessel's correction
    keepdim,  # Whether to keep reduced dimensions
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the mean
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(block_start, block_start + BLOCK_SIZE):
        if i < size:
            mean += tl.load(input_ptr + i * stride)
    mean = mean / size

    # Compute the variance
    variance = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(block_start, block_start + BLOCK_SIZE):
        if i < size:
            x = tl.load(input_ptr + i * stride)
            variance += (x - mean) ** 2
    variance = variance / (size - correction)

    # Compute the standard deviation
    std = tl.sqrt(variance)

    # Write the result to the output tensor
    if keepdim:
        tl.store(output_ptr + block_start, std)
    else:
        tl.store(output_ptr + pid, std)

def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(input.dim()))
    elif isinstance(dim, int):
        dim = (dim,)

    # Flatten the input tensor along the specified dimensions
    input_flattened = input.flatten(start_dim=0, end_dim=max(dim))

    # Compute the size and stride for the kernel
    size = input_flattened.numel()
    stride = input_flattened.stride(0)

    # Allocate the output tensor
    if out is None:
        out_shape = [1 if i in dim else s for i, s in enumerate(input.shape)] if keepdim else [s for i, s in enumerate(input.shape) if i not in dim]
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    # Launch the kernel
    grid = (triton.cdiv(size, 1024),)
    std_kernel[grid](
        input_flattened,  # Pointer to the input tensor
        out,  # Pointer to the output tensor
        size,  # Total number of elements in the input tensor
        stride,  # Stride of the input tensor
        dim,  # Dimension(s) to reduce over
        correction,  # Bessel's correction
        keepdim,  # Whether to keep reduced dimensions
        BLOCK_SIZE=1024,  # Block size for parallelization
    )

    return out
