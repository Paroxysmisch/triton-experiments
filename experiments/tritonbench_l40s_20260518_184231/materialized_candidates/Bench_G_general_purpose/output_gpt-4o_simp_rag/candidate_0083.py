import math
import torch
import triton
import triton.language as tl

# Triton kernel for row-wise quantization
@triton.jit
def _quantize_rowwise(
    x_ptr,             # Pointer to the input tensor
    output_ptr,        # Pointer to the output tensor
    output_maxs,       # Pointer to the max values tensor
    n_elements,        # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size constant
    P2: tl.constexpr,          # Power of two size constant
):
    # Calculate the block index and the element offsets within the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE

    # Load the input elements
    x = tl.load(x_ptr + offsets, mask=row_mask)

    # Calculate the absolute maximum value for normalization
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)

    # Quantize the input elements to int8
    scale = 127.0 / max_val
    output = tl.libdevice.llrint(x * scale)

    # Store the quantized output and max values
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise(x: torch.Tensor):
    # Prepare output tensors
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

    # Calculate the power of two size
    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    # Ensure CUDA compatibility
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()

    # Define grid configuration
    grid = lambda meta: (x.shape[0],)

    # Launch the Triton kernel
    _quantize_rowwise[grid](
        x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2
    )

    return output, output_maxs
