import math
import torch
import triton
import triton.language as tl

@triton.jit
def cos_func(a, b, BLOCK_SIZE: tl.constexpr):
    # Calculate offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = offset < a.shape[0]
    # Load elements from 'a' with the mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the cosine of the loaded values
    b_value = tl.cos(a_value)
    # Store the results in 'b'
    tl.store(b + offset, b_value, mask=mask)

def cos(a):
    # Create an output tensor with the same shape as 'a'
    b = torch.empty_like(a)
    # Calculate block size as the nearest power of 2 greater than sqrt(num_elements)
    BLOCK_SIZE = int(2 ** (math.ceil(math.log2(math.sqrt(a.numel())))))
    # Determine the grid size for the kernel launch
    grid = lambda meta: (triton.cdiv(a.numel(), BLOCK_SIZE),)
    # Launch the Triton kernel
    cos_func[grid](a, b, BLOCK_SIZE=BLOCK_SIZE)
    return b
