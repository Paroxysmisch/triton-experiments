import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    src_ptr,  # Pointer to source tensor
    dst_ptr,  # Pointer to destination tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Size of parallel execution block
):
    # Define the constant exponent compensator
    EXPONENT_COMPENSATOR = tl.constexpr(2.0 ** (127 - 15))
    
    # Compute the program ID and corresponding indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the tensor is not evenly divisible by BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data from source tensor
    x = tl.load(src_ptr + offsets, mask=mask)
    
    # Perform the multiplication
    y = x * EXPONENT_COMPENSATOR
    
    # Store the result in the destination tensor
    tl.store(dst_ptr + offsets, y, mask=mask)

def launch_mul_kernel(src: torch.Tensor, block_size: int = 1024):
    # Ensure input is on CUDA
    assert src.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor
    dst = torch.empty_like(src)
    
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(src.numel(), meta['BLOCK_SIZE']),)
    mul_kernel[grid](
        src_ptr=src.data_ptr(),
        dst_ptr=dst.data_ptr(),
        n_elements=src.numel(),
        BLOCK_SIZE=block_size,
    )
    
    return dst

# Example usage
if __name__ == "__main__":
    torch.set_printoptions(precision=20)
    
    # Create a sample input tensor
    src = torch.tensor([8323072], dtype=torch.int32, device='cuda').view(torch.float32)
    print('Source:', src)
    
    # Run the Triton kernel
    dst = launch_mul_kernel(src)
    print('Triton Result:', dst)
    
    # Verify with PyTorch
    dst_torch = (2.0 ** (127 - 15)) * src
    print('PyTorch Result:', dst_torch)
    
    # Check if results match
    print('Results match:', torch.allclose(dst, dst_torch, rtol=1e-5, atol=1e-5))
