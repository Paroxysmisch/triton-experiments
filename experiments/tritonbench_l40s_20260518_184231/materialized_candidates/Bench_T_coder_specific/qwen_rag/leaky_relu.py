import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    X_ptr,
    Y_ptr,
    N,
    negative_slope,
    block_size: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(N, block_size)
    X = tl.load(X_ptr + pid * block_size, mask=pid < grid_size, eviction_policy='oldest_first')
    
    # Apply LeakyReLU
    Y = tl.where(X >= 0, X, negative_slope * X)
    
    tl.store(Y_ptr + pid * block_size, Y, mask=pid < grid_size)

@triton.jit
def leaky_relu(input, negative_slope=0.01, inplace=False):
    N = input.shape[0]
    block_size = 256
    X_ptr = input.data_ptr()
    
    if inplace:
        Y_ptr = X_ptr
    else:
        Y_ptr = triton.out_ptr(input.new_empty_like(input))
    
    leaky_relu_kernel[X_grid=(N // block_size + 1,), B=(block_size,)](X_ptr, Y_ptr, N, negative_slope, block_size)
    
    if inplace:
        return input
    else:
        return input.new_tensor_from_data_ptr(Y_ptr, shape=input.shape, dtype=input.dtype)
