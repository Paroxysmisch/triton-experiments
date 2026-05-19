import triton
import triton.language as tl

@triton.jit
def relu_kernel_rank_1(x_ptr, y_ptr, n):
    # Define the grid
    pid = tl.program_id(axis=0)
    # Calculate the starting index of the current thread
    start = pid * tl.program_size(axis=0)
    # Define the mask for tl.where
    mask = start < n
    # Load data from global memory
    x = tl.load(x_ptr + start)
    # Apply ReLU operation
    y = tl.where(mask, x, 0)
    # Store the result back to global memory
    tl.store(y_ptr + start, y)

def relu_forward_wrapper_rank_1(x, y):
    # Get the pointer to the data
    x_ptr = triton.get_pointer(x)
    y_ptr = triton.get_pointer(y)
    # Calculate the number of elements
    n = x.numel()
    # Calculate the grid size
    grid = lambda:(n + tl.program_size(axis=0) - 1) // tl.program_size(axis=0),
    # Launch the kernel
    relu_kernel_rank_1[grid](x_ptr, y_ptr, n)

x = torch.randn(1000).cuda()
y = torch.empty_like(x).cuda()
relu_forward_wrapper_rank_1(x, y)
