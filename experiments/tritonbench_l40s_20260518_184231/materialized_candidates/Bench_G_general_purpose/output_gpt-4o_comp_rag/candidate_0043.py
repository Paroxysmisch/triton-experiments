import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Program ID
    pid = tl.program_id(0)
    
    # Block start index
    block_start = pid * BLOCK_SIZE
    
    # Offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices
    mask = offsets < N
    
    # Load input data
    input_data = tl.load(in_ptr + offsets, mask=mask)
    
    # Apply ReLU
    output_data = tl.where(input_data > 0, input_data, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, output_data, mask=mask)

def relu(x):
    # Ensure input tensor is on GPU
    x = x.to(device="cuda")
    
    # Output tensor
    y = torch.empty_like(x, device="cuda")
    
    # Number of elements in the input tensor
    N = x.numel()
    
    # Block size for Triton kernel
    BLOCK_SIZE = 1024
    
    # Number of blocks needed
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    relu_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return y

# Example usage
x = torch.randn(67108864, device="cuda")
y = relu(x)
print(y)
