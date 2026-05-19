import triton
import triton.language as tl
import torch

# Quantize a floating-point weight matrix into INT4 format
def quantize_int4(weights, scale):
    # Scale the weights and clip to the INT4 range
    quantized = torch.clamp(torch.round(weights / scale), -8, 7).to(torch.int8)
    return quantized

# Dequantize INT4 weights back to floating-point
def unpack_int4(quantized, scale):
    return quantized.to(torch.float32) * scale

# Triton kernel for matrix multiplication with INT4 weights
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K, scale, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    # Define the block indices
    pid = tl.program_id(0)
    block_m = pid // (N // BLOCK_SIZE)
    block_n = pid % (N // BLOCK_SIZE)

    # Define the offsets for A, B, and C
    offs_am = block_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_bn = block_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE):
        # Load A and B tiles
        a = tl.load(A + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b = tl.load(B + (k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        
        # Dequantize B (INT4 to float)
        b = b.to(tl.float32) * scale
        
        # Perform matrix multiplication
        acc += tl.dot(a, b)

    # Store the result
    tl.store(C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, acc)

# Wrapper function to set up and launch the Triton kernel
def matmul_dequantize_int4_s2(A, B_quantized, scale, M, N, K):
    # Allocate output matrix
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')

    # Launch the Triton kernel
    BLOCK_SIZE = 16
    grid = (M // BLOCK_SIZE) * (N // BLOCK_SIZE)
    matmul_kernel[grid](
        A, B_quantized, C, M, N, K, scale,
        A.stride(0), A.stride(1),
        B_quantized.stride(0), B_quantized.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return C

# Example usage
if __name__ == "__main__":
    # Define matrix dimensions
    M, N, K = 128, 128, 128

    # Create input matrices
    A = torch.randn((M, K), dtype=torch.float32, device='cuda')
    B = torch.randn((K, N), dtype=torch.float32, device='cuda')

    # Quantize B
    scale = 0.1  # Example scale factor
    B_quantized = quantize_int4(B, scale)

    # Perform matrix multiplication
    C = matmul_dequantize_int4_s2(A, B_quantized, scale, M, N, K)

    # Print the result
    print(C)
