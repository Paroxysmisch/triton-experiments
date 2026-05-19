import torch
import triton
import triton.language as tl

@triton.jit
def logit_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load the row into SRAM
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)

    # Apply clamping based on eps
    z = tl.where(row < eps, eps, row)
    z = tl.where(row > 1 - eps, 1 - eps, z)

    # Calculate logit
    logit_output = tl.where(z <= 0, float('nan'), tl.log(z / (1 - z)))

    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, logit_output, mask=col_offsets < n_cols)

def logit(input: torch.Tensor, eps: float = None, out: torch.Tensor = None) -> torch.Tensor:
    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Allocate output if not provided
    if out is None:
        out = torch.empty_like(input)

    logit_kernel[(n_rows,)](
        out,
        input,
        input.stride(0),
        out.stride(0),
        n_cols,
        eps if eps is not None else 0.0,  # Pass eps as 0.0 if None
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
