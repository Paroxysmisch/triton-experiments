import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    input_col_stride,
    reduced_size,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_row_ptr = input_ptr + row_idx * input_row_stride

    accumulator = tl.zeros((1,), dtype=tl.float32)
    for col_offset in range(0, reduced_size, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < reduced_size
        input_vals = tl.load(input_row_ptr + cols * input_col_stride, mask=mask, other=0.0)
        accumulator += tl.sum(input_vals, axis=0)

    mean_val = accumulator / numel
    output_ptr_row = output_ptr + row_idx
    tl.store(output_ptr_row, mean_val)

def mean(input, dim, keepdim=False, dtype=None, out=None):
    # Cast input to the specified dtype if provided
    if dtype is not None:
        input = input.to(dtype)
    
    # Normalize the dimensions to a sorted tuple of positive integers
    dims = dim if isinstance(dim, tuple) else (dim,)
    dims = tuple(sorted([d if d >= 0 else input.ndim + d for d in dims]))
    
    # Validate dimensions
    for d in dims:
        if d < 0 or d >= input.ndim:
            raise ValueError(f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim-1}], but got {d})")
    
    # If no reduction needed, return a clone with keepdim handled
    if len(dims) == 0:
        result = input.clone()
        if keepdim:
            new_shape = list(input.shape)
            for d in dims:
                new_shape[d] = 1
            result = result.reshape(new_shape)
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # Separate non-reduced and reduced dimensions
    non_reduced_dims = [d for d in range(input.ndim) if d not in dims]
    permute_order = non_reduced_dims + list(dims)
    permuted_input = input.permute(permute_order).contiguous()
    
    # Flatten non-reduced and reduced dimensions
    non_reduced_shape = permuted_input.shape[:len(non_reduced_dims)]
    flattened_non_reduced = 1
    for s in non_reduced_shape:
        flattened_non_reduced *= s
    reduced_size = permuted_input.numel() // flattened_non_reduced
    numel = reduced_size  # numel is product of reduced dimensions
    
    # Reshape to 2D tensor (flattened_non_reduced, reduced_size)
    flattened_input = permuted_input.reshape(flattened_non_reduced, reduced_size)
    
    # Create output tensor
    sum_output = torch.empty((flattened_non_reduced,), dtype=flattened_input.dtype, device=input.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    grid = (flattened_non_reduced,)
    mean_kernel[grid](
        flattened_input, sum_output,
        flattened_input.stride(0), flattened_input.stride(1),
        reduced_size, numel,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to non-reduced dimensions
    mean_output = sum_output.reshape(non_reduced_shape)
    
    # Insert singleton dimensions for keepdim
    if keepdim:
        for d in sorted(dims):
            mean_output = mean_output.unsqueeze(d)
    
    # Handle output tensor
    if out is not None:
        if not out.is_contiguous():
            out.copy_(mean_output)
        else:
            out.resize_(mean_output.shape)
            out.copy_(mean_output)
        return out
    else:
        return mean_output
