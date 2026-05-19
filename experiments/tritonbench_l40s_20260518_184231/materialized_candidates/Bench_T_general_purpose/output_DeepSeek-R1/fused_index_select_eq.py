import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    input_ptr,
    index_ptr,
    other_ptr,
    output_ptr,
    input_dim0, input_dim1, input_dim2,
    input_stride0, input_stride1, input_stride2,
    index_size,
    other_dim0, other_dim1, other_dim2,
    other_stride0, other_stride1, other_stride2,
    output_dim0, output_dim1, output_dim2,
    output_stride0, output_stride1, output_stride2,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = output_dim0 * output_dim1 * output_dim2
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    for idx in tl.range(start, num_elements, BLOCK_SIZE, mask=mask):
        offset = tl.load(offsers, mask=mask, other=0)
        
        # Compute 3D indices for output
        i = offset // (output_dim1 * output_dim2)
        remainder = offset % (output_dim1 * output_dim2)
        j = remainder // output_dim2
        k = remainder % output_dim2
        
        # Get index along dim
        if dim == 0:
            index_pos = i
        elif dim == 1:
            index_pos = j
        else:
            index_pos = k
        idx_val = tl.load(index_ptr + index_pos)
        
        # Compute input indices
        if dim == 0:
            i_in = idx_val
            j_in = j
            k_in = k
        elif dim == 1:
            i_in = i
            j_in = idx_val
            k_in = k
        else:
            i_in = i
            j_in = j
            k_in = idx_val
        
        input_offset = i_in * input_stride0 + j_in * input_stride1 + k_in * input_stride2
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0)
        
        # Compute other indices (assuming other is broadcasted to output shape)
        other_i = i % other_dim0
        other_j = j % other_dim1
        other_k = k % other_dim2
        other_offset = other_i * other_stride0 + other_j * other_stride1 + other_k * other_stride2
        other_val = tl.load(other_ptr + other_offset, mask=mask, other=0)
        
        # Compare and store
        res = input_val == other_val
        output_offset = i * output_stride0 + j * output_stride1 + k * output_stride2
        tl.store(output_ptr + output_offset, res, mask=mask)

def fused_index_select_eq(input: torch.Tensor, dim: int, index: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    assert index.dim() == 1, "Index must be a 1D tensor"
    assert dim < input.dim(), "dim out of range"
    
    # Compute selected tensor shape
    input_shape = list(input.shape)
    selected_shape = input_shape.copy()
    selected_shape[dim] = index.size(0)
    selected_shape = tuple(selected_shape)
    
    # Expand other to selected shape if necessary
    if isinstance(other, torch.Tensor):
        other_expanded = other.expand(selected_shape)
    else:
        other_expanded = torch.full(selected_shape, other, dtype=input.dtype, device=input.device)
    
    # Ensure input is 3D for simplicity (expand if needed)
    while input.dim() < 3:
        input = input.unsqueeze(0)
    while other_expanded.dim() < 3:
        other_expanded = other_expanded.unsqueeze(0)
    selected_shape = tuple(input.shape[:dim] + (index.size(0),) + input.shape[dim+1:])
    
    # Create output tensor
    output = torch.empty(selected_shape, dtype=torch.bool, device=input.device)
    
    # Launch kernel
    def _3d_shape(t):
        while t.dim() < 3:
            t = t.unsqueeze(-1)
        return t
    input_3d = _3d_shape(input)
    other_3d = _3d_shape(other_expanded)
    output_3d = _3d_shape(output)
    
    input_dim0, input_dim1, input_dim2 = input_3d.shape
    other_dim0, other_dim1, other_dim2 = other_3d.shape
    output_dim0, output_dim1, output_dim2 = output_3d.shape
    
    BLOCK_SIZE = 128
    num_elements = output_3d.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    fused_index_select_eq_kernel[grid](
        input_3d,
        index,
        other_3d,
        output_3d,
        input_dim0, input_dim1, input_dim2,
        input_3d.stride(0), input_3d.stride(1), input_3d.stride(2),
        index.size(0),
        other_dim0, other_dim1, other_dim2,
        other_3d.stride(0), other_3d.stride(1), other_3d.stride(2),
        output_dim0, output_dim1, output_dim2,
        output_3d.stride(0), output_3d.stride(1), output_3d.stride(2),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Handle output
    output = output.reshape(selected_shape)
    if out is not None:
        out.copy_(output)
        return out
    return output
