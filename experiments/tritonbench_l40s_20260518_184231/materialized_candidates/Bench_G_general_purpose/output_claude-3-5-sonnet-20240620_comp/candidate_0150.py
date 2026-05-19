import triton
import triton.language as tl
import torch

@triton.jit
def argmax_kernel_1(
    inp_ptr, mid_value_ptr, mid_index_ptr,
    M, BLOCK_SIZE: tl.constexpr, INT64_INDEX: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate start index for this block
    block_start = pid * BLOCK_SIZE
    
    # Load input values for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    x = tl.load(inp_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max value and index in this block
    max_val = x[0]
    max_idx = 0
    
    for i in range(1, BLOCK_SIZE):
        if i < tl.num_programs:
            curr_val = x[i]
            if curr_val > max_val:
                max_val = curr_val
                max_idx = i
    
    # Store intermediate results
    tl.store(mid_value_ptr + pid, max_val)
    if INT64_INDEX:
        tl.store(mid_index_ptr + pid, tl.int64(block_start + max_idx))
    else:
        tl.store(mid_index_ptr + pid, tl.int32(block_start + max_idx))

@triton.jit
def argmax_kernel_2(
    mid_value_ptr, mid_index_ptr, out_ptr,
    mid_size, BLOCK_MID: tl.constexpr, INT64_INDEX: tl.constexpr
):
    # Single block reduction
    pid = tl.program_id(0)
    if pid > 0:
        return
        
    # Initialize with first element
    max_val = tl.load(mid_value_ptr + 0)
    max_idx = tl.load(mid_index_ptr + 0)
    
    # Reduce across all intermediate results
    for i in range(1, mid_size):
        curr_val = tl.load(mid_value_ptr + i)
        curr_idx = tl.load(mid_index_ptr + i)
        if curr_val > max_val:
            max_val = curr_val
            max_idx = curr_idx
    
    # Store final result
    if INT64_INDEX:
        tl.store(out_ptr, tl.int64(max_idx))
    else:
        tl.store(out_ptr, tl.int32(max_idx))

@triton.jit
def argmax_kernel(
    inp_ptr, out_ptr,
    stride_dim, stride_out,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Calculate offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Create mask for valid elements
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    # Initialize max values and indices
    max_vals = tl.full([BLOCK_M, BLOCK_K], -float('inf'))
    max_idxs = tl.zeros([BLOCK_M, BLOCK_K], tl.int64 if INT64_INDEX else tl.int32)
    
    # Iterate over N dimension
    for n in range(0, N):
        # Calculate input offset
        inp_offs = offs_m[:, None] * stride_dim + n * K + offs_k[None, :]
        # Load values
        x = tl.load(inp_ptr + inp_offs, mask=mask_m[:, None] & mask_k[None, :], other=-float('inf'))
        
        # Update max values and indices
        curr_max_mask = x > max_vals
        max_vals = tl.where(curr_max_mask, x, max_vals)
        max_idxs = tl.where(curr_max_mask, n, max_idxs)
    
    # Store results
    out_offs = offs_m[:, None] * stride_out + offs_k[None, :]
    tl.store(out_ptr + out_offs, max_idxs, mask=mask_m[:, None] & mask_k[None, :])

def can_use_int32_index(size):
    return size <= 2**31 - 1

def argmax(input_tensor, dim=None):
    if dim is None:
        # Flatten input and treat as 1D
        M = input_tensor.numel()
        BLOCK_SIZE = 1024
        n_blocks = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Allocate intermediate buffers
        mid_values = torch.empty(n_blocks, dtype=torch.float32, device=input_tensor.device)
        use_int64 = not can_use_int32_index(M)
        mid_indices = torch.empty(n_blocks, 
                                dtype=torch.int64 if use_int64 else torch.int32,
                                device=input_tensor.device)
        
        # First kernel: block-wise reduction
        argmax_kernel_1[(n_blocks,)](
            input_tensor.reshape(-1), mid_values, mid_indices,
            M, BLOCK_SIZE, use_int64
        )
        
        # Second kernel: final reduction
        output = torch.empty(1, dtype=torch.int64 if use_int64 else torch.int32,
                           device=input_tensor.device)
        argmax_kernel_2[(1,)](
            mid_values, mid_indices, output,
            n_blocks, BLOCK_SIZE, use_int64
        )
        
        return output.item()
    
    else:
        # Handle specific dimension
        dim = dim if dim >= 0 else input_tensor.dim() + dim
        shape = input_tensor.shape
        
        # Calculate M, N, K
        M = 1
        for i in range(dim):
            M *= shape[i]
        N = shape[dim]
        K = 1
        for i in range(dim + 1, len(shape)):
            K *= shape[i]
        
        # Calculate strides
        stride_dim = input_tensor.stride(dim)
        
        # Configure grid and blocks
        BLOCK_M = 32
        BLOCK_K = 32
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_k = (K + BLOCK_K - 1) // BLOCK_K
        
        # Allocate output
        output_shape = list(shape)
        output_shape.pop(dim)
        output = torch.empty(output_shape,
                           dtype=torch.int64 if not can_use_int32_index(N) else torch.int32,
                           device=input_tensor.device)
        
        # Launch kernel
        argmax_kernel[(grid_m, grid_k)](
            input_tensor, output,
            stride_dim, output.stride(0) if dim > 0 else K,
            M, N, K,
            BLOCK_M, BLOCK_K,
            not can_use_int32_index(N)
        )
        
        return output
