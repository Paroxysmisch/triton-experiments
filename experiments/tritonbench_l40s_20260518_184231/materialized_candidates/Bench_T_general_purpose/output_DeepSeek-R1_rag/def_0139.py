import torch
import triton
import triton.language as tl

@triton.jit
def std_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # First pass: compute mean
    mean = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + col_offset
        mask = col_offsets < n_cols
        input_chunk = tl.load(
            input_ptr + row_start + col_offsets,
            mask=mask,
            other=0.0,
        )
        mean += tl.sum(input_chunk, axis=0)
    mean /= n_cols

    # Second pass: compute sum of squared differences
    sum_sq_diff = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + col_offset
        mask = col_offsets < n_cols
        input_chunk = tl.load(
            input_ptr + row_start + col_offsets,
            mask=mask,
            other=0.0,
        )
        diff = input_chunk - mean
        sum_sq_diff += tl.sum(diff * diff, axis=0)

    # Compute variance and standard deviation
    denominator = tl.maximum(n_cols - correction, 0)
    variance = sum_sq_diff / denominator if denominator != 0 else 0.0
    std = tl.sqrt(variance)

    # Store the result
    output_row = row_idx * output_row_stride
    tl.store(output_ptr + output_row, std)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    # Handle dim=None by reducing all dimensions
    if dim is None:
        dim = tuple(range(input.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)
    
    # Convert negative dimensions to positive
    input_ndim = input.dim()
    dim = sorted([d if d >= 0 else d + input_ndim for d in dim])
    
    # Check for duplicate dimensions
    if len(dim) != len(set(dim)):
        raise ValueError("dim has duplicate entries")
    
    # Calculate the number of elements in reduced dimensions
    reduced_prod = 1
    for d in dim:
        reduced_prod *= input.size(d)
    
    # Permute and reshape the tensor
    non_reduced_dims = [d for d in range(input_ndim) if d not in dim]
    permuted_dims = non_reduced_dims + list(dim)
    permuted_input = input.permute(permuted_dims)
    
    # Flatten non-reduced and reduced dimensions
    non_reduced_size = torch.Size(permuted_input.size()[:len(non_reduced_dims)])
    non_reduced_prod = 1
    for s in non_reduced_size:
        non_reduced_prod *= s
    flattened_input = permuted_input.contiguous().view(non_reduced_prod, -1)
    n_cols = flattened_input.size(1)
    
    # Allocate output tensor
    output = torch.empty(non_reduced_prod, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024  # Adjust based on hardware capabilities
    grid = (non_reduced_prod,)
    std_kernel[grid](
        flattened_input,
        output,
        flattened_input.stride(0),
        output.stride(0),
        n_cols,
        correction,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Determine output shape
    new_shape = []
    for d in range(input_ndim):
        if d in dim:
            if keepdim:
                new_shape.append(1)
        else:
            new_shape.append(input.size(d))
    output = output.view(new_shape)
    
    # Handle out parameter
    if out is not None:
        if not out.is_contiguous():
            raise ValueError("out tensor must be contiguous")
        out.resize_(output.shape)
        out.copy_(output)
        return out
    return output
