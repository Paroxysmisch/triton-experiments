import torch
import triton
import triton.language as tl

@triton.jit
def cos_signbit_kernel(input_ptr, cos_ptr, signbit_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program index and compute the element index
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load the input values
    input_vals = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute the cosine of each element
    cos_vals = tl.cos(input_vals)
    
    # Store the cosine results
    tl.store(cos_ptr + offsets, cos_vals, mask=offsets < n_elements)
    
    # Compute the sign bit (True for negative, False for positive or zero)
    signbit_vals = tl.bitcast(cos_vals < 0, tl.int8)
    
    # Store the sign bit results
    tl.store(signbit_ptr + offsets, signbit_vals, mask=offsets < n_elements)

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure the input is contiguous
    input = input.contiguous()
    
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Allocate output tensors for cosine results and sign bits
    cos_result = torch.empty_like(input)
    sign_bit = torch.empty_like(input, dtype=torch.int8)
    
    # Define block size for Triton kernel
    BLOCK_SIZE = 1024
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    cos_signbit_kernel[grid](input, cos_result, sign_bit, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Convert sign_bit to boolean tensor
    sign_bit_bool = sign_bit.to(torch.bool)
    
    return cos_result, sign_bit_bool

# Example usage:
# input_tensor = torch.tensor([0.0, 1.0, -1.0, 3.14159], dtype=torch.float32)
# cos_result, sign_bit = cos_signbit(input_tensor)
# print(cos_result)
# print(sign_bit)
