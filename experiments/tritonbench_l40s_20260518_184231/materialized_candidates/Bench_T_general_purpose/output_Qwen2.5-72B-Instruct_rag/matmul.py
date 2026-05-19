import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(input, other, *, out=None):
    # Ensure input and other are valid tensors
    if not isinstance(input, torch.Tensor) or not isinstance(other, torch.Tensor):
        raise ValueError("Both input and other must be tensors")

    # Determine the output shape based on input dimensions
    if input.dim() == 1 and other.dim() == 1:
        if input.size(0) != other.size(0):
            raise ValueError("Input tensors must be of the same size for dot product")
        if out is not None:
            raise ValueError("The 1D dot product version does not support an out parameter")
        return dot_product(input, other)
    elif input.dim() == 2 and other.dim() == 2:
        M, K = input.size()
        K2, N = other.size()
        if K != K2:
            raise ValueError("Incompatible dimensions for matrix-matrix product")
    elif input.dim() == 1 and other.dim() == 2:
        M = input.size(0)
        K, N = other.size()
    elif input.dim() == 2 and other.dim() == 1:
        M, K = input.size()
        N = other.size(0)
    else:
        # Handle N-dimensional tensors with broadcasting support
        M = input.size(-2)
        K = input.size(-1)
        N = other.size(-1)
        if input.size(-1) != other.size(-2):
            raise ValueError("Incompatible dimensions for batched matrix multiply")

    # Prepare the output tensor
    if out is None:
        out = torch.empty((M, N), dtype=input.dtype, device=input.device)
    else:
        if out.size() != (M, N):
            raise ValueError("Output tensor must have the correct shape")

    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        input, other, out,
        M, N, K,
        input.stride(-2), input.stride(-1),
        other.stride(-2), other.stride(-1),
        out.stride(-2), out.stride(-1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )

    return out

def dot_product(x, y):
    # Ensure x and y are 1D tensors
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Both input tensors must be 1-dimensional")
    
    if x.size(0) != y.size(0):
        raise ValueError("Input tensors must be of the same size")

    N = next_power_of_2(x.size(0))
    block_size = 1024

    # Prepare output tensor
    out = torch.empty((), dtype=torch.float32, device=x.device)
    
    # Launch Triton kernel
    grid = (1,)
    dot_product_kernel[grid](x, y, out, N, block_size)
    
    return out.item()

# Helper function to find the next power of 2
def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

# Test 1D dot product
x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
result = matmul(x, y)
print("1D Dot Product:", result)  # Expected: 32.0

# Test 2D matrix-matrix product
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
result = matmul(a, b)
print("2D Matrix-Matrix Product:\n", result)  # Expected: [[19.0, 22.0], [43.0, 50.0]]

# Test 2D matrix-vector product
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
b = torch.tensor([5.0, 6.0], device='cuda')
result = matmul(a, b)
print("2D Matrix-Vector Product:", result)  # Expected: [17.0, 39.0]

# Test 3D batched matrix multiply
a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
b = torch.tensor([[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]], device='cuda')
result = matmul(a, b)
print("3D Batched Matrix Multiply:\n", result)  # Expected: [[[19.0, 22.0], [43.0, 50.0]], [[11.0, 14.0], [25.0, 32.0]]]
