import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(input_ptr, output_ptr, n_elements):
    # Compute sigmoid for each element
    idx = tl.program_id(0)
    if idx < n_elements:
        x = tl.load(input_ptr + idx)
        sigmoid_value = 1 / (1 + tl.exp(-x))
        tl.store(output_ptr + idx, sigmoid_value)

@triton.jit
def argmax_kernel(input_ptr, output_ptr, n_elements, dim):
    # Compute argmax for the sigmoid output
    idx = tl.program_id(0)
    if idx < n_elements:
        max_idx = 0
        max_value = tl.load(input_ptr)
        for i in range(1, n_elements):
            value = tl.load(input_ptr + i)
            if value > max_value:
                max_value = value
                max_idx = i
        tl.store(output_ptr, max_idx)

import torch

def sigmoid_argmax(input: torch.Tensor, dim=None, keepdim=False) -> torch.LongTensor:
    # Apply sigmoid function
    sigmoid_output = torch.empty_like(input)
    sigmoid_kernel[(input.numel(),)](input.data_ptr(), sigmoid_output.data_ptr(), input.numel())
    
    # Compute argmax
    if dim is None:
        # Flatten the tensor and find the argmax
        max_idx = torch.argmax(sigmoid_output.flatten())
        return max_idx.unsqueeze(0) if keepdim else max_idx
    else:
        # Compute argmax along the specified dimension
        max_idx = torch.argmax(sigmoid_output, dim=dim)
        return max_idx.unsqueeze(dim) if keepdim else max_idx
