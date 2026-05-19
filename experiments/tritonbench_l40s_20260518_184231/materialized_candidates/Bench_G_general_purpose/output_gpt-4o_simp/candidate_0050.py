import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Program ID determines which block of data this kernel is responsible for
    pid = tl.program_id(0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from the input tensor
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Apply ReLU operation
    relu_x = tl.where(x > 0, x, 0.0)
    
    # Store the result in the output tensor
    tl.store(out_ptr + offsets, relu_x, mask=offsets < n_elements)

def relu(x):
    # Number of elements in the input tensor
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
