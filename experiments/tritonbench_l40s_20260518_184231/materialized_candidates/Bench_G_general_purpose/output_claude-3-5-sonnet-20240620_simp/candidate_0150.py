import triton
import triton.language as tl
import torch

@triton.jit
def argmax_kernel_1(
    input_ptr,      # Pointer to input tensor
    output_val_ptr, # Pointer to output values
    output_idx_ptr, # Pointer to output indices
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of block to process
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start/end
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Initialize max value and index
    max_value = float('-inf')
    max_index = 0
    
    # Load and process elements within bounds
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find max in block
    for i in range(BLOCK_SIZE):
        if i < n_elements - start_idx:
            if x[i] > max_value:
                max_value = x[i]
                max_index = start_idx + i
    
    # Store results
    tl.store(output_val_ptr + pid, max_value)
    tl.store(output_idx_ptr + pid, max_index)

@triton.jit
def argmax_kernel_2(
    values_ptr,     # Pointer to block max values
    indices_ptr,    # Pointer to block indices
    out_idx_ptr,    # Final output index
    n_blocks,       # Number of blocks to process
    BLOCK_SIZE: tl.constexpr,
):
    # Single thread to find global max
    if tl.program_id(0) == 0:
        max_value = float('-inf')
        max_index = 0
        
        # Compare across all blocks
        for i in range(n_blocks):
            curr_val = tl.load(values_ptr + i)
            curr_idx = tl.load(indices_ptr + i)
            
            if curr_val > max_value:
                max_value = curr_val
                max_index = curr_idx
        
        # Store final result
        tl.store(out_idx_ptr, max_index)

@triton.jit
def argmax_kernel(
    input_ptr,      # Input tensor
    output_ptr,     # Output indices
    stride,         # Stride of dimension to reduce
    size,           # Size of dimension to reduce
    other_dims,     # Product of other dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_out = tl.program_id(0)  # Output position
    pid_in = tl.program_id(1)   # Input chunk
    
    # Compute base pointers
    in_base = pid_out + pid_in * stride * other_dims
    
    # Initialize max tracking
    max_val = float('-inf')
    max_idx = 0
    
    # Process elements
    for i in range(0, size, BLOCK_SIZE):
        # Load block
        offs = in_base + (i + tl.arange(0, BLOCK_SIZE)) * stride
        mask = i + tl.arange(0, BLOCK_SIZE) < size
        x = tl.load(input_ptr + offs, mask=mask, other=float('-inf'))
        
        # Update max
        for j in range(BLOCK_SIZE):
            if i + j < size and x[j] > max_val:
                max_val = x[j]
                max_idx = i + j
    
    # Store result
    if pid_in == 0:
        tl.store(output_ptr + pid_out, max_idx)

def argmax(input_tensor, dim=None, output_dtype=torch.int64):
    # Handle full tensor argmax
    if dim is None:
        n_elements = input_tensor.numel()
        BLOCK_SIZE = 1024
        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        
        # Allocate intermediate storage
        block_values = torch.empty(n_blocks, device=input_tensor.device, dtype=input_tensor.dtype)
        block_indices = torch.empty(n_blocks, device=input_tensor.device, dtype=output_dtype)
        output = torch.empty((), device=input_tensor.device, dtype=output_dtype)
        
        # Launch kernels
        argmax_kernel_1[(n_blocks,)](
            input_tensor.data_ptr(),
            block_values.data_ptr(),
            block_indices.data_ptr(),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        argmax_kernel_2[(1,)](
            block_values.data_ptr(),
            block_indices.data_ptr(),
            output.data_ptr(),
            n_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output
    
    # Handle dimension-specific argmax
    else:
        # Get dimension info
        dim = dim if dim >= 0 else input_tensor.dim() + dim
        size = input_tensor.size(dim)
        stride = input_tensor.stride(dim)
        
        # Calculate other dimensions
        other_dims = 1
        for i, s in enumerate(input_tensor.shape):
            if i != dim:
                other_dims *= s
        
        # Prepare output
        output_shape = list(input_tensor.shape)
        output_shape.pop(dim)
        output = torch.empty(output_shape, device=input_tensor.device, dtype=output_dtype)
        
        # Launch kernel
        BLOCK_SIZE = min(128, size)
        grid = (other_dims, triton.cdiv(size, BLOCK_SIZE))
        
        argmax_kernel[grid](
            input_tensor.data_ptr(),
            output.data_ptr(),
            stride,
            size,
            other_dims,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output
