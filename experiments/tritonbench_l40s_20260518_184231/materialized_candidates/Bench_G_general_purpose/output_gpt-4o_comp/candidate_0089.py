import torch
import triton
import triton.language as tl

# Triton kernel for row-wise quantization
@triton.jit
def _quantize_rowwise(
    x_ptr,               # Pointer to the input tensor
    output_ptr,          # Pointer to the output tensor (quantized values)
    output_maxs,         # Pointer to the tensor storing max values per row
    n_elements,          # Total number of elements in the output tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
    P2: tl.constexpr          # Power of 2 ceiling of the row size
):
    # Compute the row index and the starting position for this block
    row_idx = tl.program_id(0)
    row_start = row_idx * P2

    # Load the row into shared memory
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to avoid out-of-bounds accesses
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute the maximum absolute value in the row
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)

    # Store the max value for this row
    if tl.thread_idx() == 0:
        tl.store(output_maxs + row_idx, max_val)

    # Scale the elements of the row to fit into the int8 range [-128, 127]
    scale = 127.0 / max_val if max_val > 0 else 0.0
    quantized = tl.extra.cuda.libdevice.llrint(x * scale)

    # Store the quantized values in the output tensor
    tl.store(output_ptr + offsets, quantized, mask=mask)


# Wrapper function for row-wise quantization
def quantize_rowwise(input_tensor, block_size=128):
    """
    Wrapper function for the _quantize_rowwise kernel.

    Args:
        input_tensor (torch.Tensor): 2D input tensor to quantize (CUDA tensor).
        block_size (int): Block size for Triton kernel (default: 128).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and max values tensor.
    """
    assert input_tensor.is_cuda, "Input tensor must be a CUDA tensor"
    assert input_tensor.ndim == 2, "Input tensor must be 2D"

    # Input tensor dimensions
    num_rows, num_cols = input_tensor.shape

    # Power of 2 ceiling for the number of columns
    p2 = 2 ** ((num_cols - 1).bit_length())

    # Allocate output tensors
    quantized_output = torch.empty_like(input_tensor, dtype=torch.int8)
    max_values = torch.empty(num_rows, dtype=torch.float32, device='cuda')

    # Grid size corresponds to the number of rows
    grid = (num_rows,)

    # Launch the Triton kernel
    _quantize_rowwise[grid](
        input_tensor,
        quantized_output,
        max_values,
        input_tensor.numel(),
        BLOCK_SIZE=block_size,
        P2=p2
    )

    return quantized_output, max_values

# Example usage
input_tensor = torch.randn(1024, 256, device='cuda')  # 2D input tensor
quantized_tensor, max_values = quantize_rowwise(input_tensor)

print("Quantized Tensor:", quantized_tensor)
print("Max Values:", max_values)
