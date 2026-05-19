import torch
import triton
import triton.language as tl
import math

# Kernel 1: Find local maximum and its index within blocks
@triton.jit
def argmax_kernel_1(
    inp_ptr,  # Input tensor pointer
    val_ptr,  # Intermediate values pointer
    idx_ptr,  # Intermediate indices pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of each block
    INT64_INDEX: tl.constexpr = False,  # Whether to use 64-bit indices
):
    # Get program ID and handle index type
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    
    # Calculate offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(inp_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find maximum value and its index in this block
    max_val = tl.max(x)
    max_idx = tl.argmax(x)
    
    # Store results
    tl.store(val_ptr + pid, max_val)
    tl.store(idx_ptr + pid, pid * BLOCK_SIZE + max_idx)

# Kernel 2: Aggregate results from first kernel
@triton.jit
def argmax_kernel_2(
    val_ptr,  # Intermediate values pointer
    idx_ptr,  # Intermediate indices pointer
    out_val_ptr,  # Output value pointer
    out_idx_ptr,  # Output index pointer
    n_blocks,  # Number of blocks to process
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    # Load values and indices
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks
    
    values = tl.load(val_ptr + offsets, mask=mask, other=-float('inf'))
    indices = tl.load(idx_ptr + offsets, mask=mask, other=0)
    
    # Find global maximum
    max_val = tl.max(values)
    max_idx = indices[tl.argmax(values)]
    
    # Store final results
    tl.store(out_val_ptr, max_val)
    tl.store(out_idx_ptr, max_idx)

# Kernel 3: Handle multi-dimensional case
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def argmax_kernel(
    inp_ptr,  # Input tensor pointer
    out_idx_ptr,  # Output indices pointer
    M,  # Outer dimension size
    N,  # Inner dimension size (dimension to reduce)
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    INT64_INDEX: tl.constexpr = False,  # Whether to use 64-bit indices
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
        
    # Calculate row indices
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = row_idx < M
    
    # Initialize tracking variables
    max_vals = tl.full([BLOCK_M, 1], -float('inf'), dtype=tl.float32)
    max_idxs = tl.zeros([BLOCK_M, 1], dtype=tl.int64 if INT64_INDEX else tl.int32)
    
    # Process blocks along N dimension
    for n_start in range(0, N, BLOCK_N):
        col_idx = n_start + tl.arange(0, BLOCK_N)[None, :]
        col_mask = col_idx < N
        mask = row_mask & col_mask
        
        # Load block
        block_ptr = inp_ptr + row_idx * N + col_idx
        x = tl.load(block_ptr, mask=mask, other=-float('inf'))
        
        # Update maximum values and indices
        curr_max = tl.max(x, axis=1)[:, None]
        curr_idx = n_start + tl.argmax(x, axis=1)[:, None]
        
        # Update if new maximum is found
        update_mask = curr_max > max_vals
        max_vals = tl.where(update_mask, curr_max, max_vals)
        max_idxs = tl.where(update_mask, curr_idx, max_idxs)
    
    # Store results
    out_ptr = out_idx_ptr + row_idx[:, 0]
    tl.store(out_ptr, max_idxs[:, 0], mask=row_mask)

def argmax(input_tensor, dim=None, keepdim=False):
    if dim is None:
        # Flatten tensor and find global argmax
        n_elements = input_tensor.numel()
        block_size = triton.next_power_of_2(min(2048, n_elements))
        n_blocks = triton.cdiv(n_elements, block_size)
        
        # Allocate intermediate storage
        device = input_tensor.device
        dtype = input_tensor.dtype
        use_int64 = n_elements >= (2**31)
        idx_dtype = torch.int64 if use_int64 else torch.int32
        
        temp_values = torch.empty(n_blocks, dtype=dtype, device=device)
        temp_indices = torch.empty(n_blocks, dtype=idx_dtype, device=device)
        
        # Run kernels
        argmax_kernel_1[(n_blocks,)](
            input_tensor.reshape(-1), temp_values, temp_indices,
            n_elements, block_size, INT64_INDEX=use_int64
        )
        
        final_value = torch.empty(1, dtype=dtype, device=device)
        final_index = torch.empty(1, dtype=idx_dtype, device=device)
        
        argmax_kernel_2[(1,)](
            temp_values, temp_indices, final_value, final_index,
            n_blocks, min(n_blocks, 1024)
        )
        
        return final_index.item()
    
    else:
        # Handle reduction along specific dimension
        if isinstance(dim, int):
            dim = [dim]
        dim = [d if d >= 0 else d + input_tensor.ndim for d in dim]
        
        # Reshape tensor for reduction
        shape = list(input_tensor.shape)
        n_reduce = 1
        for d in dim:
            n_reduce *= shape[d]
            shape[d] = 1
        
        M = input_tensor.numel() // n_reduce
        use_int64 = M >= (2**31) or n_reduce >= (2**31)
        
        # Allocate output
        idx_dtype = torch.int64 if use_int64 else torch.int32
        output = torch.empty(shape, dtype=idx_dtype, device=input_tensor.device)
        
        # Run kernel
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        argmax_kernel[grid](
            input_tensor, output, M, n_reduce,
            INT64_INDEX=use_int64
        )
        
        if not keepdim:
            output = output.squeeze(dim=dim)
        
        return output
