import triton
import triton.language as tl

@triton.jit
def fused_repeat_interleave_log_softmax_kernel(input_ptr, repeats_ptr, output_ptr, input_size, repeats_size, dim, BLOCK_SIZE: tl.constexpr):
    # Calculate the global index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < input_size

    # Load input and repeats
    input_data = tl.load(input_ptr + idx[mask])
    repeats = tl.load(repeats_ptr)

    # Repeat interleave operation
    repeated_data = tl.zeros((BLOCK_SIZE * max(repeats)), dtype=input_data.dtype)
    for i in range(repeats_size):
        for j in range(repeats[i]):
            repeated_data[i * max(repeats) + j] = input_data[i]

    # Log-softmax activation
    max_val = tl.max(repeated_data)
    exp_data = tl.exp(repeated_data - max_val)
    log_softmax_data = tl.log(exp_data / tl.sum(exp_data, dim=dim))

    # Store the result
    tl.store(output_ptr + idx[mask], log_softmax_data)

import torch
import triton
import triton.language as tl

def fused_repeat_interleave_log_softmax(input: torch.Tensor, repeats: torch.Tensor, dim=None, *, output_size=None, dtype=None, out=None) -> torch.Tensor:
    # Validate input shapes and types
    if dim is None:
        input = input.flatten()
    
    input_size = input.numel()
    repeats_size = repeats.numel()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty(output_size if output_size is not None else input_size, dtype=dtype if dtype is not None else input.dtype)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(input_size, 1024),)
    fused_repeat_interleave_log_softmax_kernel[grid](input_ptr=input.data_ptr(), 
                                                      repeats_ptr=repeats.data_ptr(), 
                                                      output_ptr=out.data_ptr(), 
                                                      input_size=input_size, 
                                                      repeats_size=repeats_size, 
                                                      dim=dim)
    
    return out
