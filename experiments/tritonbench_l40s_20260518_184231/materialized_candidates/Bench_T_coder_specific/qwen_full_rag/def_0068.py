import torch
import triton
import triton.language as tl


@triton.jit
def add_mean_kernel(input, other, out, n_elements, D, BLOCK_SIZE : tl.constexpr,  HAS_ALPHA : tl.constexpr, HAS_DIM : tl.constexpr, KEEPDIM : tl.constexpr, IS_TENSOR_OTHER : tl.constexpr, dtype : tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if HAS_DIM :
        dim_offsets = offsets % D
        offset_input = offsets - dim_offsets
        offset_other = dim_offsets 
    else :
        offset_input = offsets 
        offset_other = 0
        
    if IS_TENSOR_OTHER :
        other_value = tl.load(other + offset_other, mask = mask)
    else :
        other_value = other 
        
    if HAS_ALPHA :
        other_value = other_value * alpha 
        
    input_val = tl.load(input + offset_input, mask = mask)  
    result = (input_val + other_value).to(dtype) / (1+HAS_ALPHA)
    
    if KEEPDIM :
        tl.store(out + offset_input ,result, mask = mask)
    else :
        store_address = offset_input // D 
        tl.store(out + store_address,result, mask = mask)


def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    """
    Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes the mean value along the specified dimension.
    If no dimension is specified, it computes the mean over all elements. Supports broadcasting, type promotion, and works with integer, float, and complex inputs.
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        dim (int or tuple of ints, optional): The dimension(s) to reduce. Default: None.
        alpha (Number, optional): The multiplier for other. Default: 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default: False.
        dtype (torch.dtype, optional): The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation. This is useful for preventing data type overflows. Default: None.
        out (Tensor, optional): The output tensor.
    Returns:
        Tensor: The resultant tensor after addition and mean computation.
    """
    if dtype is None : 
        dtype = input.dtype 
    
    if out is not None : 
        assert out.dtype == dtype 
    else : 
        out = torch.empty_like(input, dtype=dtype) 
    
    input = input.contiguous()
    if isinstance(other, torch.Tensor) :
        other = other.contiguous() 
        IS_TENSOR_OTHER = True 
    else : 
        IS_TENSOR_OTHER = False 
    
    if dim is None :
        result = torch.add(input.mean(), other*alpha)
        if not keepdim : 
            out = out.reshape(input.shape)
        out.copy_(result.expand_as(out))
        return out 

    input_dim = list(range(0, input.dim()))
    if isinstance(dim, int) : 
        dim = [dim]
        
    for d in sorted(dim, reverse=True):
        assert d >= -input.ndim and d < input.ndim, "Invalid dim"
        del input_dim[d]
    
    input_dim_strides = list(input.stride())
    for d in input_dim : 
        input_dim_strides[d] = 0 
        
    n_elements = input.numel()
    D = abs(torch.prod(torch.tensor([input.size(d) for d in dim])).item())
    ndim_out = len(input.shape) - len(dim)
    out_shape = list(input.shape[:])
    for i, d in enumerate(dim) : 
        out_shape[d] = 1 
    out_view = out.view(out_shape)
    out_view_stride = out_view.stride()
    out_view_dim = list(range(0, out_view.dim()))
    for d in sorted(dim, reverse=True):
        del out_view_dim[d]
        
    out_view_dim_strides = list(out_view.stride())
    for d in out_view_dim : 
        out_view_dim_strides[d] = 0 
    m = out_view.numel()
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = lambda meta : (triton.cdiv(m, meta['BLOCK_SIZE']),)
    
    with torch.cuda.device(input.device):
        add_mean_kernel[grid](input, other, out_view, n_elements, D, BLOCK_SIZE, HAS_ALPHA=(alpha!=1), HAS_DIM=True, KEEPDIM=keepdim, IS_TENSOR_OTHER=IS_TENSOR_OTHER, dtype=dtype)
    
    return out
