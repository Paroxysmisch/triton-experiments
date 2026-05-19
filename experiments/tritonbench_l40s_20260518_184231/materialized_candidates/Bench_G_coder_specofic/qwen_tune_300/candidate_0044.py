import torch
import triton
import triton.language as tl
from triton import next_power_of_2

@triton.jit
def softmax_kernel(
    input_ptr: tl.tensor,  # Pointer to the input data
    output_ptr: tl.tensor, # Pointer to the output data
    stride_input_row: int, # Stride for accessing rows in input
    stride_output_row: int,# Stride for accessing rows in output
    BLOCK_SIZE: tl.constexpr # Block size for triton kernel
):
    row_index = tl.program_id(0) # Get the current row index
    input_row_ptr = input_ptr + row_index * stride_input_row # Pointer to the current row in input
    output_row_ptr = output_ptr + row_index * stride_output_row # Pointer to the current row in output
    
    # Load the row into SRAM
    x = tl.load(input_row_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Compute softmax
    max_val = tl.max(x)
    x_minus_max = x - max_val
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator)
    softmax_output = numerator / denominator
    
    # Write back the result to DRAM
    tl.store(output_row_ptr + tl.arange(0, BLOCK_SIZE), softmax_output)

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        device = x.device
        dtype = x.dtype
        x = x.contiguous()

        # Configure kernel parameters
        grid = (x.shape[0],)
        BLOCK_SIZE = next_power_of_2(x.shape[1])
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        # Prepare output tensor
        y = torch.empty_like(x)

        # Launch Triton kernel
        softmax_kernel[grid](
            x,
            y,
            x.stride(0),
            y.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return y
