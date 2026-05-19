import torch
import triton
import triton.language as tl

@triton.jit
def sample_from_prob_kernel(x_ptr,  # *Pointer* to first input vector.
                            output_ptr,  # *Pointer* to output vector.
                            n_elements,  # Size of the vector.
                            BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                            ):
    # ... existing code ...

def sample_from_prob(x: torch.Tensor):
    # Prepare the output tensor
    output = torch.empty_like(x)
    n_elements = output.numel()
    # Define the grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # Call the Triton function
    sample_from_prob_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
