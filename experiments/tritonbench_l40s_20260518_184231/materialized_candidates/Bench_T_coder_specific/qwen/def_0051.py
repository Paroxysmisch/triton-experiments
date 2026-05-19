import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr, # Pointer to the output tensor
    n,          # Number of elements in the input tensor
    kernel_size,# Size of the pooling window
    stride,     # Stride of the pooling window
    padding,    # Zero-padding added to both sides of the input
    ceil_mode,  # If True, uses ceil instead of floor to compute the output shape
    count_include_pad, # If True, includes the zero-padding in the averaging calculation
    BLOCK_SIZE: tl.constexpr, # Block size
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(n, BLOCK_SIZE)

    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = i < n

    # Add padding
    padded_input = tl.zeros_like(input_ptr)
    padded_input[padding:i + padding] = input_ptr[i]

    # Compute cosine
    cos_values = tl.cos(padded_input)

    # Initialize output
    output = tl.zeros_like(output_ptr)

    # Compute pool indices
    pool_start = i // stride
    pool_end = pool_start + kernel_size
    pool_valid = pool_end <= (n + padding)

    # Perform pooling
    if pool_valid:
        pool_indices = tl.arange(pool_start, pool_end)
        pool_values = cos_values[pool_indices]
        pool_count = tl.sum(pool_valid)
        if count_include_pad:
            pool_count += kernel_size
        avg_value = tl.sum(pool_values) / pool_count
        output[i] = avg_value

    output_ptr[i] = output[i]

# Define the wrapper function
def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    if stride is None:
        stride = kernel_size

    n = input.numel()
    output = torch.empty((n,), device=input.device, dtype=input.dtype)

    # Launch the Triton kernel
    block_size = 256
    num_blocks = (n + block_size - 1) // block_size
    cos_avg_pool1d_kernel[input.nbytes(), block_size](input.data_ptr(), output.data_ptr(), n, kernel_size, stride, padding, ceil_mode, count_include_pad)

    return output
