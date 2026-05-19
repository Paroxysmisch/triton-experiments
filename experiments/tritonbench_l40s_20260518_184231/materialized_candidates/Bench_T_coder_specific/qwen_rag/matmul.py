import triton
import triton.language as tl
import torch

# Define a helper function to calculate the next power of 2
def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

# Define the Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, z_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    row = pid // num_blocks_n
    col = pid % num_blocks_n
    block_row_start = row * BLOCK_SIZE_M
    block_col_start = col * BLOCK_SIZE_N
    m = block_row_start + tl.arange(0, BLOCK_SIZE_M)
    n = block_col_start + tl.arange(0, BLOCK_SIZE_N)
    k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        A = tl.load(x_ptr + (m[:, None] * K + k_block * BLOCK_SIZE_K + k)[None, :], mask=(m[:, None] < M) & (k_block * BLOCK_SIZE_K + k < K), boundary_check=False)
        B = tl.load(y_ptr + (k_block * BLOCK_SIZE_K + k) * N + n, mask=(k_block * BLOCK_SIZE_K + k < K) & (n < N), boundary_check=False)
        accumulator += A * B

    output_mask = (m[:, None] < M) & (n < N)
    tl.store(z_ptr + (m[:, None] * N + n), accumulator, mask=output_mask)

# Define the wrapper function for matrix multiplication
def matmul(x, y, out=None):
    # Determine the dimensions of the input tensors
    x_shape = x.shape
    y_shape = y.shape
    
    # Check the number of dimensions
    if len(x_shape) == 1 and len(y_shape) == 1:
        # Dot product
        if x_shape[0] != y_shape[0]:
            raise ValueError("Input tensors must be of the same size for dot product")
        N = next_power_of_2(x_shape[0])
        block_size = 1024
        
        # Prepare output tensor
        out = torch.empty((), dtype=torch.float32, device=x.device)
        
        # Launch Triton kernel
        grid = (1,)
        dot_product_kernel[grid](x, y, out, N, block_size)
        
        return out.item()
    elif len(x_shape) == 2 and len(y_shape) == 2:
        # Matrix-matrix product
        M, K = x_shape
        K2, N = y_shape
        if K != K2:
            raise ValueError("Inner dimensions must match for matrix-matrix product")
        
        # Prepare output tensor
        out = torch.empty((M, N), dtype=torch.float32, device=x.device)
        
        # Launch Triton kernel
        block_size_m = 32
        block_size_n = 32
        block_size_k = 32
        grid = (tl.cdiv(M, block_size_m) * tl.cdiv(N, block_size_n),)
        matmul_kernel[grid](x, y, out, M, N, K, block_size_m, block_size_n, block_size_k)
        
        return out
    elif len(x_shape) == 2 and len(y_shape) == 1:
        # Matrix-vector product
        M, K = x_shape
        if K != y_shape[0]:
            raise ValueError("Inner dimensions must match for matrix-vector product")
        
        # Prepare output tensor
        out = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        # Launch Triton kernel
        block_size_m = 32
        block_size_n = 1
        block_size_k = 32
        grid = (tl.cdiv(M, block_size_m),)
        matmul_kernel[grid](x, y, out, M, 1, K, block_size_m, block_size_n, block_size_k)
        
        return out
    else:
        raise NotImplementedError("Unsupported tensor dimensions for matmul")

# Example usage
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda')
    y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device='cuda')
    print(matmul(x, y))  # Should print 32.0

    x = torch.randn(3, 4, dtype=torch.float32, device='cuda')
    y = torch.randn(4, 5, dtype=torch.float32, device='cuda')
    print(matmul(x, y))  # Should print a 3x5 matrix
