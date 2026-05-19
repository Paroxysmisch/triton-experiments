import triton
import triton.language as tl
import torch

# Triton kernel for dequantization
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}),
    ],
    key=['K', 'N']
)
@triton.jit
def dequantize_kernel(b_ptr, b_scale_ptr, fpb_ptr, K, N, stride_bk, stride_bn, stride_bscale_k, stride_fpbk, stride_fpbn, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Define the block indices
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Compute the start of the block
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create pointers for the current block
    b_offset = block_start_k * stride_bk + block_start_n * stride_bn
    b_scale_offset = block_start_k * stride_bscale_k
    fpb_offset = block_start_k * stride_fpbk + block_start_n * stride_fpbn

    # Load the block of b and b_scale
    b = tl.load(b_ptr + b_offset, mask=(block_start_n < N) & (block_start_k < K), other=0).to(tl.float32)
    b_scale = tl.load(b_scale_ptr + b_scale_offset, mask=(block_start_k < K), other=0).to(tl.float32)

    # Dequantize
    fpb = b * b_scale[:, None]

    # Store the result
    tl.store(fpb_ptr + fpb_offset, fpb, mask=(block_start_n < N) & (block_start_k < K))

# Wrapper function
def matmul_dequantize_int8(a, b, b_scale, K, N):
    # Check matrix dimensions
    assert a.shape[1] == K, "Matrix a's second dimension must match K"
    assert b.shape == (K, N), "Matrix b must have dimensions K x N"
    assert b_scale.shape == (K,), "b_scale must have dimension K"

    # Allocate output tensor
    fpb = torch.empty((K, N), dtype=torch.float32, device=a.device)

    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K']))
    dequantize_kernel[grid](b, b_scale, fpb, K, N, b.stride(0), b.stride(1), b_scale.stride(0), fpb.stride(0), fpb.stride(1))

    # Perform matrix multiplication using PyTorch
    c = torch.mm(a, fpb)

    return c

# Example usage
K, N, M = 512, 256, 128
a = torch.randn(M, K, dtype=torch.float32, device='cuda')
b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')
b_scale = torch.rand(K, dtype=torch.float32, device='cuda')

c = matmul_dequantize_int8(a, b, b_scale, K, N)
print(c)
