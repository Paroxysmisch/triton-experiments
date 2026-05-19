import torch
import triton
import triton.language as tl

# Triton kernel function
@triton.jit
def sample_from_prob_kernel(x_ptr,  # *Pointer* to first input vector.
                            output_ptr,  # *Pointer* to output vector.
                            n_elements,  # Size of the vector.
                            BLOCK_SIZE: tl.constexpr):  # Number of elements each program should process.
  
  # Identify the program ID
  pid = tl.program_id(axis=0)  # Using a 1D launch grid, axis is 0
  
  # Calculate offsets and create a mask for out-of-bounds access
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements

  # Load data from DRAM, masked for out-of-bounds
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x * 2

  # Store the processed data back to DRAM
  tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function
def sample_from_prob(x: torch.Tensor):
  # Preallocate the output tensor
  output = torch.empty_like(x)
  n_elements = output.numel()

  # Define the grid size for launching the kernel
  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

  # Launch the kernel
  sample_from_prob_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

  # Return the output tensor
  return output
