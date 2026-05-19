import triton
import triton.language as tl

# Triton kernel to compute the mean along specified dimensions
@triton.jit
def mean_dim_kernel(X, M, N, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute program ID
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Compute the range of rows and columns this program will process
    row_start = pid * BLOCK_M
    row_end = min(row_start + BLOCK_M, M)
    
    # Initialize the sum and count for the mean calculation
    sum_val = tl.zeros((BLOCK_N,), dtype=tl.float32)
    count = 0
    
    # Iterate over the rows assigned to this program
    for row in range(row_start, row_end):
        # Load the row from the tensor
        row_data = tl.load(X + row * N + tl.arange(0, BLOCK_N))
        # Accumulate the sum
        sum_val += row_data
        count += 1
    
    # Compute the mean
    mean_val = sum_val / count
    
    # Store the result in the output tensor
    tl.store(out + tl.arange(0, BLOCK_N), mean_val)

# Wrapper function to handle input transformation and kernel invocation
def mean_dim(X, dim, BLOCK_M, BLOCK_N):
    # Get the shape of the input tensor
    M, N = X.shape
    
    # Check if the specified dimension is valid
    if dim < 0 or dim >= len(X.shape):
        raise ValueError("Invalid dimension")
    
    # Allocate the output tensor
    if dim == 0:
        out_shape = (1, N)
    else:
        out_shape = (M, 1)
    out = triton.empty(out_shape, dtype=triton.float32, device=X.device)
    
    # Compute the number of programs needed
    num_programs = (M + BLOCK_M - 1) // BLOCK_M
    
    # Launch the kernel
    mean_dim_kernel[(num_programs,)](X, M, N, out, BLOCK_M, BLOCK_N)
    
    return out

# Example usage
import torch

# Create a tensor
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')

# Compute the mean along dimension 0
mean_dim_0 = mean_dim(X, dim=0, BLOCK_M=1, BLOCK_N=3)
print("Mean along dimension 0:", mean_dim_0)

# Compute the mean along dimension 1
mean_dim_1 = mean_dim(X, dim=1, BLOCK_M=2, BLOCK_N=1)
print("Mean along dimension 1:", mean_dim_1)
