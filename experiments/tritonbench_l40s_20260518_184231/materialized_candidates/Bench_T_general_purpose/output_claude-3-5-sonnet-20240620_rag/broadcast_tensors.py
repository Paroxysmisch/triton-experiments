import triton
import triton.language as tl
import torch
from typing import List

@triton.jit
def _broadcast_kernel(output_ptr, input_ptrs, output_shape, num_tensors, num_elements):
    # Each thread computes one element of the output tensor
    idx = tl.program_id(0)
    if idx >= num_elements:
        return

    # Calculate the indices for each tensor based on broadcasting rules
    indices = [0] * num_tensors
    for i in range(num_tensors):
        indices[i] = idx % output_shape[i]
        idx //= output_shape[i]

    # Load values from each input tensor and compute the output
    output_value = 0.0
    for i in range(num_tensors):
        input_ptr = input_ptrs[i]
        output_value += tl.load(input_ptr + indices[i])

    # Store the result in the output tensor
    tl.store(output_ptr + idx, output_value)

def broadcast_tensors(*tensors: torch.Tensor) -> List[torch.Tensor]:
    # Ensure all tensors are of the same type
    assert all(tensors[0].dtype == t.dtype for t in tensors), "All tensors must be of the same type."

    # Get the broadcasted shape
    broadcast_shape = torch.broadcast_shapes(*(t.shape for t in tensors))

    # Create output tensor
    output_tensors = [torch.empty(broadcast_shape, dtype=tensors[0].dtype, device=tensors[0].device) for _ in tensors]

    # Prepare input pointers for the kernel
    input_ptrs = [t.contiguous().data_ptr() for t in tensors]
    output_ptrs = [out.data_ptr() for out in output_tensors]

    # Launch the kernel
    num_elements = torch.prod(torch.tensor(broadcast_shape)).item()
    grid = (num_elements,)
    _broadcast_kernel[grid](output_ptrs, input_ptrs, broadcast_shape, len(tensors), num_elements)

    # Return the broadcasted tensors
    return output_tensors
