import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 128}, num_stages=1, num_warps=4),
    ],
    key=['n_elements', 'c', 'num_groups']
)
def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, out=None):
    n_elements = input1.shape[0]
    c = input1.shape[-1]
    
    if out is None:
        out = input1.new_empty(input1.shape)
    
    block_size_x = 32
    grid_size_x = (n_elements + block_size_x - 1) // block_size_x
    
    fused_add_mul_groupnorm_kernel[
        grid_size_x, block_size_x
    ](
        input1.data_ptr(), input2.data_ptr(), weight.data_ptr(), bias.data_ptr(),
        out.data_ptr(), out.data_ptr(),
        n_elements, c, num_groups, eps,
        input1.stride(0), input2.stride(0), out.stride(0), out.stride(0)
    )
    
    return out
