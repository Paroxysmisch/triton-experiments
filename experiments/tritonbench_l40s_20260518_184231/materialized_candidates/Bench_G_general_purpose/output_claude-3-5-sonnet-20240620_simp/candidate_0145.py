import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    input_ptr, scale_ptr, output_ptr,
    M, N,
    stride_im, stride_in, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 2D grid
    grid_m = pid // (N // BLOCK_SIZE_N)
    grid_n = pid % (N // BLOCK_SIZE_N)
    
    # Starting indices
    offm = grid_m * BLOCK_SIZE_M
    offn = grid_n * BLOCK_SIZE_N
    
    # Pointers to the input and output
    input_ptr = input_ptr + offm * stride_im + offn * stride_in
    output_ptr = output_ptr + offm * stride_om + offn * stride_on
    
    # Load the scale factor
    scale = tl.load(scale_ptr)
    
    # Load and dequantize
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if offm + i < M and offn + j < N:
                x = tl.load(input_ptr + i * stride_im + j * stride_in)
                y = tl.cast(x, tl.float32) * scale
                tl.store(output_ptr + i * stride_om + j * stride_on, y)

def matmul_dequantize_int8(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda and scale.is_cuda, "All tensors must be on GPU"
    assert b.dtype == torch.int8, "Tensor b must be int8"
    assert a.dtype == torch.float32, "Tensor a must be float32"
    assert scale.dtype == torch.float32, "Scale must be float32"
    assert scale.numel() == 1, "Scale must be a single value"
    
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} and {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Allocate temporary buffer for dequantized b
    b_float = torch.empty((K, N), device=b.device, dtype=torch.float32)
    
    # Grid for dequantization
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch dequantization kernel
    dequantize_kernel[grid](
        b, scale, b_float,
        K, N,
        b.stride(0), b.stride(1), b_float.stride(0), b_float.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    
    # Perform matrix multiplication
    c = torch.matmul(a, b_float)
    
    return c

# Example usage
if __name__ == "__main__":
    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)
    scale = torch.tensor([0.1], device='cuda', dtype=torch.float32)
    
    result = matmul_dequantize_int8(a, b, scale)
    print(result.shape)  # Should be (M, N)
