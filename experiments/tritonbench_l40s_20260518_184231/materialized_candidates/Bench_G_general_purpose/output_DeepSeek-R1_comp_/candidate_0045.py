import triton
import triton.language as tl
import torch

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    # Pointers to tensors
    input_ptr,
    output_ptr,
    # Scalar exponent
    exponent,
    # Tensor dimensions
    num_elts,
    # Strides
    stride_x,
    stride_y,
    # Tile size (static)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID for current block
    pid = tl.program_id(0)
    
    # Create block pointers with boundary checks
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(num_elts,),
        strides=(stride_x,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(num_elts,),
        strides=(stride_y,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    
    # Load data with automatic boundary masking
    x = tl.load(input_block_ptr, boundary_check=(0,))
    # Compute element-wise power
    y = tl.pow(x, exponent)
    # Store results with boundary checks
    tl.store(output_block_ptr, y, boundary_check=(0,))

def pow_func_scalar_tensor_wrapper_rank_1(x, exponent, out=None):
    # Validate inputs and allocate output if needed
    if out is None:
        out = torch.empty_like(x)
    assert x.shape == out.shape, "Input/output shape mismatch"
    assert x.is_cuda and out.is_cuda, "Tensors must be on GPU"
    
    # Extract tensor metadata
    def get_meta(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.data_ptr(), tensor.stride(0)
        return tensor.data, tensor.strides[0]
    
    x_ptr, x_stride = get_meta(x)
    out_ptr, out_stride = get_meta(out)
    num_elts = x.size(0)
    
    # Heuristic configuration
    def next_pow2(n): return 1 << (n - 1).bit_length()
    
    BLOCK_SIZE = max(128, min(1024, next_pow2(num_elts)))
    num_warps = BLOCK_SIZE // 32
    
    # Compute grid dimensions
    grid = (triton.cdiv(num_elts, BLOCK_SIZE),)
    
    # Launch kernel with selected configuration
    pow_func_scalar_tensor_kernel_rank_1[grid](
        x_ptr, out_ptr, exponent, num_elts, x_stride, out_stride,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    
    return out
