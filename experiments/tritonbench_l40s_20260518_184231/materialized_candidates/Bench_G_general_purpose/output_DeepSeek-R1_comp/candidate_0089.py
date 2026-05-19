import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # Get the current row index
    pid = tl.program_id(0)
    n_rows = tl.num_programs(0)
    k = n_elements // n_rows  # Number of elements per row

    # Calculate the starting pointers for the current row
    row_offset = pid * k
    x_row = x_ptr + row_offset
    out_row = output_ptr + row_offset
    max_ptr = output_maxs + pid

    # Initialize max_abs to zero
    max_abs = 0.0

    # First pass: Compute the maximum absolute value in the row
    for i in range(0, P2, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < k
        x = tl.load(x_row + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        current_max = tl.max(abs_x, axis=0)
        max_abs = tl.maximum(max_abs, current_max)

    # Compute the scaling factor to fit into int8
    scale = 127.0 / max_abs

    # Second pass: Quantize and store the values
    for i in range(0, P2, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < k
        x = tl.load(x_row + offsets, mask=mask, other=0.0)
        quantized = tl.extra.cuda.libdevice.llrint(x * scale)
        tl.store(out_row + offsets, quantized, mask=mask)

    # Store the maximum absolute value for the row
    tl.store(max_ptr, max_abs)

def quantize_rowwise(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be a CUDA tensor"
    assert x.dim() == 2, "Input tensor must be 2-dimensional"

    M, K = x.shape
    n_elements = M * K

    # Compute the next power of two greater than or equal to K
    P2 = 1
    while P2 < K:
        P2 <<= 1

    # Set block size (can be tuned for performance)
    BLOCK_SIZE = 128

    # Ensure BLOCK_SIZE divides P2 to avoid partial loops
    if P2 % BLOCK_SIZE != 0:
        P2 = ((P2 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    # Allocate output tensors
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(M, device=x.device, dtype=torch.float32)

    # Launch kernel with appropriate grid size
    grid = (M,)
    _quantize_rowwise[grid](
        x, output, output_maxs, n_elements,
        BLOCK_SIZE=BLOCK_SIZE, P2=P2
    )

    return output, output_maxs
