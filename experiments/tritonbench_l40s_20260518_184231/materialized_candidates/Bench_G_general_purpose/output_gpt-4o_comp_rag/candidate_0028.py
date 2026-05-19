import triton
import triton.language as tl
import torch

# Define the kernel
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_o_batch, stride_o_m, stride_o_n,
    M, N, K,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_K: tl.constexpr
):
    # Define program ids
    pid = tl.program_id(axis=0)
    
    # Compute the start of the tile
    batch_id = pid // GROUP_M
    tile_id = pid % GROUP_M
    tile_m = tile_id * TILE_M
    tile_n = batch_id * TILE_N
    
    # Create pointers for A and B tiles
    A_tile_ptr = A_ptr + tile_m * stride_a_m + tile_id * TILE_K * stride_a_k
    B_tile_ptr = B_ptr + tile_n * stride_b_n + tile_id * TILE_K * stride_b_k
    
    # Create pointers for the output tile
    O_tile_ptr = O_ptr + tile_m * stride_o_m + tile_n * stride_o_n
    
    # Create accumulators
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, TILE_K):
        # Load tiles of A and B
        A_tile = tl.load(A_tile_ptr + k * stride_a_k, mask=True)
        B_tile = tl.load(B_tile_ptr + k * stride_b_k, mask=True)
        
        # Perform matrix multiplication and accumulate
        acc += tl.dot(A_tile, B_tile)
    
    # Store the result in the output tensor
    tl.store(O_tile_ptr, acc, mask=True)

# Define the wrapper function
def bmm(A, B):
    # Get the shapes of the input tensors
    batch, M, K = A.shape
    _, _, N = B.shape
    
    # Allocate output tensor
    O = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
    
    # Define tile sizes and group size
    TILE_M, TILE_N, TILE_K = 32, 32, 32
    GROUP_M = 8
    
    # Launch the kernel
    grid = (batch * GROUP_M,)
    bmm_kernel[grid](
        A, B, O,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        M, N, K,
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
        GROUP_M=GROUP_M,
        DIVISIBLE_M=(M % TILE_M == 0), DIVISIBLE_N=(N % TILE_N == 0), DIVISIBLE_K=(K % TILE_K == 0)
    )
    
    return O

# Example usage
A = torch.randn(16, 128, 64, device='cuda', dtype=torch.float32)
B = torch.randn(16, 64, 128, device='cuda', dtype=torch.float32)
O = bmm(A, B)
