import triton
import triton.language as tl
import torch

@triton.jit
def _quantize_rowwise_kernel(
    x_ptr, 
    output_ptr, 
    output_maxs_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr, 
    P2: tl.constexpr
):
    row_id = tl.program_id(0)
    row_offset = row_id * P2

    # -----------------------------
    # 1) Compute row-wise maximum
    # -----------------------------
    max_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # We'll traverse the row in chunks of BLOCK_SIZE
    for start in range(0, P2, BLOCK_SIZE):
        offsets = row_offset + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.where(
            mask, 
            tl.load(x_ptr + offsets, mask=mask, other=0.0), 
            0.0
        )
        abs_values = tl.abs(values)
        max_acc = tl.maximum(max_acc, abs_values)
    row_max = tl.max(max_acc, axis=0)

    # -----------------------------
    # 2) Quantize and write output
    # -----------------------------
    # Avoid division by zero
    inv_scale = tl.where(row_max > 0, 127.0 / row_max, 0.0)

    for start in range(0, P2, BLOCK_SIZE):
        offsets = row_offset + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.where(
            mask,
            tl.load(x_ptr + offsets, mask=mask, other=0.0),
            0.0
        )
        scaled_vals = values * inv_scale
        # Round to nearest int
        quantized = tl.libdevice.llrint(scaled_vals)
        # Clamp to [-127, 127]
        clamped = tl.minimum(tl.maximum(quantized, -127), 127)
        tl.store(output_ptr + offsets, clamped.to(tl.int8), mask=mask)
        
    # -----------------------------
    # 3) Write max value of the row
    # -----------------------------
    tl.store(
        output_maxs_ptr + row_id, 
        row_max
    )


def quantize_rowwise(input_tensor: torch.Tensor):
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    rows, cols = input_tensor.shape
    # Next power of 2 for column dimension
    p2_cols = 1
    while p2_cols < cols:
        p2_cols <<= 1

    # Flatten input for easy pointer arithmetic
    x_flat = input_tensor.contiguous().flatten()
    n_elements = x_flat.numel()

    # Create output tensors
    output = torch.empty_like(input_tensor, dtype=torch.int8, device=input_tensor.device)
    output_maxs = torch.empty(rows, dtype=torch.float32, device=input_tensor.device)

    # Grid is based on the number of rows
    grid = (rows,)

    _quantize_rowwise_kernel[grid](
        x_flat, 
        output, 
        output_maxs, 
        n_elements, 
        BLOCK_SIZE=1024, 
        P2=p2_cols
    )

    return output, output_maxs
