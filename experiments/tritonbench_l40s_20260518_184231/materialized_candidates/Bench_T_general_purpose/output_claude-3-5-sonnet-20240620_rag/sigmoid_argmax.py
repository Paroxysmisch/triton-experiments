import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(output_ptr, input_ptr, n_elements):
    # Compute the sigmoid function for each element
    idx = tl.program_id(0)
    if idx < n_elements:
        x = tl.load(input_ptr + idx)
        sigmoid_value = 1 / (1 + tl.exp(-x))
        tl.store(output_ptr + idx, sigmoid_value)

def sigmoid_argmax(input: torch.Tensor, dim=None, keepdim=False) -> torch.LongTensor:
    n_elements = input.numel()
    output = torch.empty_like(input)

    # Launch the sigmoid kernel
    sigmoid_kernel[(n_elements,)](output, input, n_elements)

    # Compute the argmax
    if dim is None:
        # Flatten the output tensor and find the index of the maximum value
        max_index = torch.argmax(output)
        return max_index.unsqueeze(0) if keepdim else max_index
    else:
        # Compute argmax along the specified dimension
        max_indices = torch.argmax(output, dim=dim)
        if keepdim:
            return max_indices.unsqueeze(dim)
        return max_indices
