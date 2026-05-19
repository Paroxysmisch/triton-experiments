import triton
import triton.language as tl
import torch

# Triton kernel to compute the reciprocal of each element in the input tensor
@triton.jit
def reciprocal_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program index
    pid = tl.program_id(0)
    # Create a range of indices for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to handle cases where the number of elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    # Load data from the input tensor
    input_data = tl.load(input_ptr + offsets, mask=mask)
    # Compute the reciprocal
    result = 1.0 / input_data
    # Store the result in the output tensor
    tl.store(output_ptr + offsets, result, mask=mask)

# Wrapper function
def reciprocal(input, *, out=None):
    # Promote integral inputs to the default scalar type (usually float32)
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        input = input.to(torch.get_default_dtype())
    
    # Determine the number of elements
    n_elements = input.numel()
    
    # Create an output tensor if none is provided
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, 1024),)
    reciprocal_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out

# Verification
input_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
output_tensor = reciprocal(input_tensor)
print(output_tensor)  # Expected: tensor([1.0000, 0.5000, 0.3333, 0.2500], dtype=torch.float32)
