import triton
import triton.language as tl
import numpy as np

# Define the Triton kernel
@triton.jit
def matmul_tma_load_store(
    A_ptr, B_ptr, C_ptr,
    A_shape, B_shape, C_shape,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUTPUT_F16: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(A_shape[0], BLOCK_M)
    grid_n = tl.cdiv(B_shape[1], BLOCK_N)
    grid_k = tl.cdiv(B_shape[0], BLOCK_K)
    
    # Matrix A and B are transposed
    A_base_ptr = tl.make_block_ptr(
        A_ptr,
        shape=(A_shape[0], A_shape[1]),
        strides=(A_shape[1], 1),
        offsets=(pid // grid_k * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        block_strides=(A_shape[1], 1),
        meta=meta
    )
    
    B_base_ptr = tl.make_block_ptr(
        B_ptr,
        shape=(B_shape[0], B_shape[1]),
        strides=(B_shape[1], 1),
        offsets=(0, pid % grid_k * BLOCK_K),
        block_shape=(BLOCK_K, BLOCK_N),
        block_strides=(B_shape[1], 1),
        meta=meta
    )
    
    C_base_ptr = tl.make_block_ptr(
        C_ptr,
        shape=(C_shape[0], C_shape[1]),
        strides=(C_shape[1], 1),
        offsets=(pid // grid_n * BLOCK_M, pid % grid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        block_strides=(C_shape[1], 1),
        meta=meta
    )
    
    # Initialize the output block
    C_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(grid_k):
        A_block = tl.load(A_base_ptr)
        B_block = tl.load(B_base_ptr)
        
        # Perform the matrix multiplication
        C_block += tl.dot(A_block, B_block)
        
        # Move to the next block of B
        B_base_ptr.x += BLOCK_K
    
    # Store the result
    if OUTPUT_F16:
        C_block = C_block.to(tl.float16)
    
    tl.store(C_base_ptr, C_block)

# Define the wrapper function
def wrapper_tma_load_store(
    M, N, K,
    num_warps=2, num_ctas=1,
    transpose_A=False, transpose_B=False,
    output_format='float32'
):
    # Generate random matrices A and B
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    
    # Transpose matrices if required
    if transpose_A:
        A = A.T
    if transpose_B:
        B = B.T
    
    # Allocate matrix C
    C = np.zeros((M, N), dtype=np.float32)
    
    # Convert matrices to Triton tensors
    A_tensor = tl.tensor(A, dtype=tl.float32)
    B_tensor = tl.tensor(B, dtype=tl.float32)
    C_tensor = tl.tensor(C, dtype=tl.float32)
    
    # Launch the Triton kernel
    grid = (num_ctas,)
    block = (128,)
    config = triton.Config(
        instantiation_shapes=[(M, N, K)],
        grid=grid,
        block=block,
        num_warps=num_warps
    )
    
    # Call the Triton kernel
    matmul_tma_load_store(
        A_tensor, B_tensor, C_tensor,
        A.shape, B.shape, C.shape,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        OUTPUT_F16=output_format == 'float16',
        meta=meta
    )
    
    # Convert the result back to a NumPy array
    C = C_tensor.numpy()
    
    return C
