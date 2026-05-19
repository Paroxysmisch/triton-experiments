import torch
import triton
import triton.language as tl

@triton.jit
def tensordot_kernel(
    a_ptr, b_ptr, c_ptr,
    a_strides, b_strides, c_strides,
    a_shape, b_shape, c_shape,
    dims_a, dims_b, num_dims,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr, ACC_TYPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    
    # Compute the grid and block indices
    grid_m = c_shape[0] // BLOCK_SIZE_M
    grid_n = c_shape[1] // BLOCK_SIZE_N
    grid_k = c_shape[2] // BLOCK_SIZE_K
    
    # Compute the block indices
    block_m = pid % grid_m
    block_n = (pid // grid_m) % grid_n
    block_k = (pid // (grid_m * grid_n)) % grid_k
    
    # Compute the block start indices
    rm = block_m * BLOCK_SIZE_M
    rn = block_n * BLOCK_SIZE_N
    rk = block_k * BLOCK_SIZE_K
    
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)
    
    # Iterate over the contraction dimension
    for k in range(0, c_shape[2], BLOCK_SIZE_K):
        # Load the input tiles
        a_tile = tl.load(a_ptr + rm * a_strides[0] + rk * a_strides[1] + tl.arange(0, BLOCK_SIZE_M)[:, None] * a_strides[0] + tl.arange(0, BLOCK_SIZE_K)[None, :])
        b_tile = tl.load(b_ptr + rk * b_strides[0] + rn * b_strides[1] + tl.arange(0, BLOCK_SIZE_K)[:, None] * b_strides[0] + tl.arange(0, BLOCK_SIZE_N)[None, :])
        
        # Perform the matrix multiplication
        acc += tl.dot(a_tile, b_tile)
    
    # Store the result
    c_tile = tl.load(c_ptr + rm * c_strides[0] + rn * c_strides[1] + tl.arange(0, BLOCK_SIZE_M)[:, None] * c_strides[0] + tl.arange(0, BLOCK_SIZE_N)[None, :])
    c_tile += acc
    tl.store(c_ptr + rm * c_strides[0] + rn * c_strides[1] + tl.arange(0, BLOCK_SIZE_M)[:, None] * c_strides[0] + tl.arange(0, BLOCK_SIZE_N)[None, :], c_tile)

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    # Convert dims to a list of lists if it's an integer
    if isinstance(dims, int):
        dims = (list(range(-dims, 0)), list(range(dims)))
    
    # Ensure dims is a tuple of two lists
    if isinstance(dims, list):
        dims = (dims[0], dims[1])
    
    # Validate the dimensions
    if len(dims[0]) != len(dims[1]):
        raise ValueError("The number of dimensions to contract must be the same for both tensors.")
    
    # Get the shapes and strides
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    a_strides = list(a.stride())
    b_strides = list(b.stride())
    
    # Compute the output shape
    c_shape = []
    for i in range(len(a_shape)):
        if i not in dims[0]:
            c_shape.append(a_shape[i])
    for i in range(len(b_shape)):
        if i not in dims[1]:
            c_shape.append(b_shape[i])
    
    # Create the output tensor
    c = torch.empty(c_shape, dtype=a.dtype, device=a.device)
    c_strides = list(c.stride())
    
    # Compute the number of dimensions
    num_dims = len(dims[0])
    
    # Launch the kernel
    grid = (c_shape[0] * c_shape[1],)
    tensordot_kernel[grid](
        a, b, c,
        a_strides, b_strides, c_strides,
        a_shape, b_shape, c_shape,
        dims[0], dims[1], num_dims,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16,
        GROUP_SIZE_M=8, SPLIT_K=1, ACC_TYPE=tl.float32
    )
    
    return c
