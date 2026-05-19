import triton
import triton.language as tl
import torch
import math

# Triton Kernel
@triton.jit
def cos_func(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Program ID determines which block of data this kernel is processing
    pid = tl.program_id(0)
    
    # Compute the offset for this block
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we only process valid indices
    mask = offset < n_elements
    
    # Load input values from tensor `a` using the offset and mask
    a_value = tl.load(a_ptr + offset, mask=mask)
    
    # Compute the cosine of each element
    b_value = tl.cos(a_value)
    
    # Store the results in tensor `b` using the offset and mask
    tl.store(b_ptr + offset, b_value, mask=mask)

# Python Wrapper Function
def cos(a: torch.Tensor):
    # Ensure input tensor is on the GPU
    assert a.is_cuda, "Input tensor `a` must be on GPU."
    
    # Get the number of elements in the input tensor
    n_elements = a.numel()
    
    # Determine the BLOCK_SIZE as the nearest power of 2 greater than sqrt(n_elements)
    BLOCK_SIZE = 2 ** math.ceil(math.log2(math.sqrt(n_elements)))
    
    # Ensure BLOCK_SIZE does not exceed the number of elements
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Triton typically supports block sizes up to 1024
    
    # Calculate the number of blocks (grid size)
    grid = (math.ceil(n_elements / BLOCK_SIZE),)
    
    # Allocate output tensor `b` on the GPU
    b = torch.empty_like(a)
    
    # Launch the Triton kernel
    cos_func[grid](a_ptr=a.data_ptr(),
                   b_ptr=b.data_ptr(),
                   n_elements=n_elements,
                   BLOCK_SIZE=BLOCK_SIZE)
    
    return b

# Example Usage
if __name__ == "__main__":
    # Input tensor on GPU
    a = torch.linspace(0, 2 * math.pi, 10000, device="cuda")
    
    # Compute cosine using the Triton kernel
    b = cos(a)
    
    # Validate results with PyTorch's cosine function
    torch.testing.assert_close(b, torch.cos(a))
    print("Cosine computation is correct!")
