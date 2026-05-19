import triton
import triton.language as tl
import torch

@triton.jit
def broadcast_kernel(
    # Pointers to input and output tensors
    out_ptr, in_ptr,
    # Original shape information
    in_stride_0, in_stride_1,
    out_stride_0, out_stride_1,
    # Output dimensions
    out_dim_0, out_dim_1,
    # Original dimensions
    in_dim_0, in_dim_1,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate output indices
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    out_row = offsets // out_dim_1
    out_col = offsets % out_dim_1
    
    # Mask for valid output indices
    mask = (out_row < out_dim_0) & (out_col < out_dim_1)
    
    # Map output indices to input indices with broadcasting
    in_row = out_row % in_dim_0
    in_col = out_col % in_dim_1
    
    # Calculate input offsets
    in_offsets = in_row * in_stride_0 + in_col * in_stride_1
    out_offsets = out_row * out_stride_0 + out_col * out_stride_1
    
    # Load and store with broadcasting
    x = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + out_offsets, x, mask=mask)

def broadcast_tensors(*tensors):
    if not tensors:
        return []
    
    # Get maximum number of dimensions
    max_dims = max(tensor.dim() for tensor in tensors)
    
    # Pad shapes with ones for broadcasting
    padded_shapes = []
    for tensor in tensors:
        shape = list(tensor.shape)
        while len(shape) < max_dims:
            shape.insert(0, 1)
        padded_shapes.append(shape)
    
    # Calculate output shape
    output_shape = []
    for dims in zip(*padded_shapes):
        output_shape.append(max(dims))
    
    # Create output tensors
    outputs = []
    for tensor in tensors:
        # Create output tensor with broadcast shape
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Grid and block sizes
        BLOCK_SIZE = 1024
        num_elements = output.numel()
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        # Launch kernel for each tensor
        broadcast_kernel[grid](
            output.data_ptr(),
            tensor.data_ptr(),
            tensor.stride(-2) if tensor.dim() > 1 else 0,
            tensor.stride(-1) if tensor.dim() > 0 else 0,
            output.stride(-2) if output.dim() > 1 else 0,
            output.stride(-1) if output.dim() > 0 else 0,
            output.shape[-2] if output.dim() > 1 else 1,
            output.shape[-1] if output.dim() > 0 else 1,
            tensor.shape[-2] if tensor.dim() > 1 else 1,
            tensor.shape[-1] if tensor.dim() > 0 else 1,
            BLOCK_SIZE
        )
        outputs.append(output)
    
    return outputs
