import triton
import triton.language as tl
import torch

# Define the block size
BLOCK_SIZE = 1024

@triton.jit
def cos_func(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the block index
    block_idx = tl.program_id(axis=0)
    
    # Compute the starting offset for this block
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we only process valid elements
    mask = offset < n_elements
    
    # Load elements from the input tensor 'a'
    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    
    # Compute the cosine of each element
    b = tl.cos(a)
    
    # Store the results in the output tensor 'b'
    tl.store(b_ptr + offset, b, mask=mask)

def cos(A):
    # Ensure input is a contiguous tensor
    A = A.contiguous()
    
    # Create the output tensor B with the same shape as A
    B = torch.empty_like(A)
    
    # Calculate the number of elements in the input tensor
    n_elements = A.numel()
    
    # Calculate the number of blocks needed
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the Triton kernel
    cos_func[grid_size](A, B, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return B

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    A = torch.randn(2048, device='cuda')
    
    # Compute the cosine using the Triton kernel
    B = cos(A)
    
    # Verify the result with PyTorch's cosine function
    assert torch.allclose(B, torch.cos(A), atol=1e-6), "Results do not match!"
