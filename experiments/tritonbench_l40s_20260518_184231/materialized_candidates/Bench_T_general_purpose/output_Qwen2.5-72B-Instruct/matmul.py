import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    input_ptr,  # Pointer to the first input tensor
    other_ptr,  # Pointer to the second input tensor
    output_ptr, # Pointer to the output tensor
    input_stride_m, input_stride_n,  # Strides for the input tensor
    other_stride_m, other_stride_n,  # Strides for the other tensor
    output_stride_m, output_stride_n, # Strides for the output tensor
    M, N, K,  # Dimensions of the matrices
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Compute the block ID in the grid
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    pid_m = (pid % num_pid_in_group) // num_pid_n
    pid_n = (pid % num_pid_in_group) % num_pid_n

    # Compute the block bounds
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N
    rk = 0

    # Compute the block bounds for the input and other tensors
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, K),
        strides=(input_stride_m, input_stride_n),
        offsets=(rm, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    other_block_ptr = tl.make_block_ptr(
        base=other_ptr,
        shape=(K, N),
        strides=(other_stride_m, other_stride_n),
        offsets=(0, rn),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1)
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_m, output_stride_n),
        offsets=(rm, rn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )

    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the input and other blocks
        input_block = tl.load(input_block_ptr)
        other_block = tl.load(other_block_ptr)

        # Perform the matrix multiplication
        accumulator += tl.dot(input_block, other_block)

        # Advance the block pointers
        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_SIZE_K))
        other_block_ptr = tl.advance(other_block_ptr, (BLOCK_SIZE_K, 0))

    # Store the result to the output tensor
    tl.store(output_block_ptr, accumulator.to(tl.float16))

import torch
import triton
import triton.language as tl

def matmul(input, other, *, out=None):
    # Determine the dimensions of the input tensors
    input_shape = input.shape
    other_shape = other.shape

    # Determine the output shape
    if len(input_shape) == 1 and len(other_shape) == 1:
        # 1D dot product
        if out is not None:
            raise ValueError("1D dot product does not support an out parameter.")
        return torch.dot(input, other)
    elif len(input_shape) == 2 and len(other_shape) == 2:
        # 2D matrix-matrix product
        M, K = input_shape
        K, N = other_shape
    elif len(input_shape) == 2 and len(other_shape) == 1:
        # 2D matrix-vector product
        M, K = input_shape
        N = 1
    elif len(input_shape) > 2 or len(other_shape) > 2:
        # Batched matrix multiply with broadcasting support
        input = input.view(-1, input_shape[-2], input_shape[-1])
        other = other.view(-1, other_shape[-2], other_shape[-1])
        M, K = input.shape[-2:]
        N = other.shape[-1]
    else:
        raise ValueError("Unsupported input dimensions.")

    # Determine the output tensor
    if out is None:
        out = torch.empty((input_shape[0], N), dtype=input.dtype, device=input.device)

    # Determine the strides for the input, other, and output tensors
    input_stride_m, input_stride_k = input.stride()[-2:]
    other_stride_k, other_stride_n = other.stride()[-2:]
    output_stride_m, output_stride_n = out.stride()[-2:]

    # Define the grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch the kernel
    matmul_kernel[grid](
        input.data_ptr(), other.data_ptr(), out.data_ptr(),
        input_stride_m, input_stride_k,
        other_stride_k, other_stride_n,
        output_stride_m, output_stride_n,
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return out

# Test 1: 1D dot product
a = torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda')
b = torch.tensor([4, 5, 6], dtype=torch.float32, device='cuda')
result = matmul(a, b)
print(result)  # Expected: 32.0

# Test 2: 2D matrix-matrix product
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cuda')
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device='cuda')
result = matmul(a, b)
print(result)  # Expected: [[19, 22], [43, 50]]

# Test 3: 2D matrix-vector product
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cuda')
b = torch.tensor([5, 6], dtype=torch.float32, device='cuda')
result = matmul(a, b)
print(result)  # Expected: [17, 39]

# Test 4: Batched matrix multiply
a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32, device='cuda')
b = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=torch.float32, device='cuda')
result = matmul(a, b)
print(result)  # Expected: [[[29, 32], [67, 74]], [[127, 136], [203, 214]]]
