import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_xn, stride_y,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(stride_xm, BLOCK_M)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    # Define a block pointer to manage memory access
    block_ptr_x = tl.make_block_ptr(
        base=x_ptr,
        shape=(stride_xm, stride_xn),
        strides=(stride_xm, stride_xn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1)
    )

    # Load a block of data
    x = tl.load(block_ptr_x)

    # Compute the row-wise maxima
    row_max = tl.max(x, axis=1)

    # Store the output
    y_ptr[pid_m * stride_y + pid_n] = row_max[pid_m]

def load_reduce(x, y):
    # Get the shape of the input matrix
    stride_xm, stride_xn = x.shape
    stride_y = y.shape[0]

    # Launch the Triton kernel
    BLOCK_M = 32
    BLOCK_N = 32
    grid_m = tl.cdiv(stride_xm, BLOCK_M)
    grid_n = tl.cdiv(stride_xn, BLOCK_N)
    grid_size = grid_m * grid_n

    # Create Triton kernel arguments
    x_ptr = x.data_ptr()
    y_ptr = y.data_ptr()

    # Launch the kernel
    load_reduce_kernel[grid_size, (BLOCK_M, BLOCK_N, 1)](
        x_ptr, y_ptr,
        stride_xm, stride_xn, stride_y,
        BLOCK_M, BLOCK_N
    )

    # Convert the output to a NumPy array for comparison
    y_np = y.cpu().numpy()

    # Compare the result with PyTorch's max function
    torch_result = torch.max(x, dim=1).values.cpu().numpy()
    assert_close(y_np, torch_result, rtol=1e-5, atol=1e-5)

def assert_close(a, b, rtol=1e-5, atol=1e-5):
    assert np.allclose(a, b, rtol=rtol, atol=atol), f"Arrays are not close: {a} vs {b}"

# Example usage
x = torch.randn(100, 50, device='cuda')
y = torch.empty(100, device='cuda')
load_reduce(x, y)
