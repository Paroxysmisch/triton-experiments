import triton
import triton.language as tl
import torch
from typing import Union, Tuple

@triton.jit
def adaptive_avg_pool2d_sigmoid_kernel(
    input_ptr, output_ptr,
    in_height, in_width,
    out_height, out_width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Calculate the pooling window size
    h_start = pid // out_width * in_height // out_height
    h_end = (pid // out_width + 1) * in_height // out_height
    w_start = pid % out_width * in_width // out_width
    w_end = (pid % out_width + 1) * in_width // out_width
    
    # Initialize sum for average pooling
    sum_val = tl.float32(0)
    count = tl.float32(0)
    
    # Perform adaptive average pooling
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            idx = h * in_width + w
            sum_val += tl.load(input_ptr + idx, mask=True)
            count += 1
    
    # Compute average
    avg_val = sum_val / count
    
    # Apply sigmoid function
    sigmoid_val = 1 / (1 + tl.exp(-avg_val))
    
    # Store the result
    tl.store(output_ptr + pid, sigmoid_val)


def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Determine output dimensions
    if isinstance(output_size, int):
        out_height = out_width = output_size
    else:
        out_height, out_width = output_size

    # Input dimensions
    in_height, in_width = input.shape[-2], input.shape[-1]

    # Allocate output tensor
    output = torch.empty((out_height, out_width), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (out_height * out_width,)
    adaptive_avg_pool2d_sigmoid_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE=1024  # Assuming a block size, can be tuned based on hardware
    )

    return output
