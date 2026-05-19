import triton
import triton.language as tl
import torch

# Define the Triton kernel for dropout
@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    # Calculate the program's grid index
    pid = tl.program_id(0)
    
    # Calculate the range of indices this program will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from x_ptr and mask from x_keep_ptr
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    mask = tl.load(x_keep_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Apply dropout: scale non-zero entries by 1/(1-p)
    scale = 1.0 / (1.0 - p)
    dropout_result = tl.where(mask, x * scale, 0.0)
    
    # Store the result back to output_ptr
    tl.store(output_ptr + offsets, dropout_result, mask=offsets < n_elements)

# Wrapper function for dropout
def dropout(x, mask, p, BLOCK_SIZE=1024):
    assert x.shape == mask.shape, "Input and mask must have the same shape"
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Get number of elements
    n_elements = x.numel()
    
    # Calculate the number of blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    _dropout[grid](x, mask, output, n_elements, p, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Create input tensor and dropout mask
    x = torch.rand(4096, device='cuda')
    mask = (torch.rand(4096, device='cuda') > 0.5).float()  # Example mask with p=0.5
    
    # Apply dropout
    p = 0.5
    output = dropout(x, mask, p)
    
    print("Input:", x)
    print("Mask:", mask)
    print("Output:", output)
