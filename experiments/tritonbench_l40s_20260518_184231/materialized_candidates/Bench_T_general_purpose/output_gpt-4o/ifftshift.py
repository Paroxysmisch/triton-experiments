import torch
import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(input_ptr, output_ptr, size, stride, n_dims, offsets, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = idx < size
    input_idx = idx.clone()
    
    # Compute new indices after the shift
    for i in range(n_dims):
        dim_size = offsets[i * 2 + 1]
        dim_offset = offsets[i * 2]
        stride_dim = stride[i]
        
        # Determine the new position after ifftshift
        shift = dim_size // 2
        input_idx = tl.where(input_idx % dim_size < shift, 
                             input_idx + shift, 
                             input_idx - shift)
    
    # Compute flat index
    flat_idx = tl.sum(input_idx * stride, axis=0)
    
    # Load input and store to output
    output = tl.load(input_ptr + flat_idx, mask=mask)
    tl.store(output_ptr + idx, output, mask=mask)

def ifftshift(input, dim=None):
    if dim is None:
        dim = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dim = (dim,)

    # Prepare strides and offsets
    strides = [input.stride(d) for d in dim]
    offsets = []
    for d in dim:
        size = input.size(d)
        shift = size // 2
        offsets.extend([shift, size])
    
    # Flatten the input and output for easier indexing
    input_flat = input.flatten()
    output_flat = torch.empty_like(input_flat)
    
    # Launch Triton kernel
    size = input_flat.numel()
    BLOCK_SIZE = 1024  # Adjust as needed for optimal performance
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    ifftshift_kernel[grid](input_flat, output_flat, size, strides, len(dim), offsets, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape the output back to the original input shape
    output = output_flat.view(input.shape)
    return output

# Example usage
input_tensor = torch.fft.fftshift(torch.arange(16).reshape(4, 4))
output_tensor = ifftshift(input_tensor)
print(output_tensor)
