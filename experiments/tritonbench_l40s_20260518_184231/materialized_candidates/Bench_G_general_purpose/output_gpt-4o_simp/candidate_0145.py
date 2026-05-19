import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    int8_ptr, scale_ptr, float_ptr, 
    n_elements, BLOCK_SIZE: tl.constexpr
):
    # Program ID determines which block of data this instance will handle
    pid = tl.program_id(0)
    
    # Define the range of indices this block will handle
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load int8 values and scale factor
    int8_vals = tl.load(int8_ptr + offsets, mask=offsets < n_elements, other=0).to(tl.float32)
    scale = tl.load(scale_ptr)
    
    # Dequantize
    float_vals = int8_vals * scale
    
    # Store result
    tl.store(float_ptr + offsets, float_vals, mask=offsets < n_elements)


import torch

def matmul_dequantize_int8(a, int8_matrix, scale):
    # Ensure the int8_matrix is on the correct device
    assert int8_matrix.dtype == torch.int8
    assert a.device == int8_matrix.device
    
    # Get dimensions
    n, k = a.shape
    k_, m = int8_matrix.shape
    assert k == k_, "Incompatible dimensions for matrix multiplication"

    # Allocate memory for the dequantized matrix
    float_matrix = torch.empty((k, m), device=int8_matrix.device, dtype=torch.float32)

    # Dequantize the int8 matrix
    grid = lambda meta: (triton.cdiv(k * m, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024  # Adjust based on your GPU's capabilities
    dequantize_kernel[grid](int8_matrix, scale, float_matrix, k * m, BLOCK_SIZE=BLOCK_SIZE)
    
    # Perform matrix multiplication
    c = torch.matmul(a, float_matrix)
    
    return c

# Example usage
a = torch.randn(64, 128, device='cuda', dtype=torch.float32)
int8_matrix = torch.randint(-128, 127, (128, 256), device='cuda', dtype=torch.int8)
scale = torch.tensor(0.1, device='cuda', dtype=torch.float32)

c = matmul_dequantize_int8(a, int8_matrix, scale)
