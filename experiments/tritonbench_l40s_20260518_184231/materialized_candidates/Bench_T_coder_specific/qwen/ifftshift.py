import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(
    x_ptr,
    o_ptr,
    n_elements,
    stride,
    num_dims,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    grid_stride = tl.cdiv(n_elements, block_size)

    for i in range(block_size):
        global_idx = block_start + i
        if global_idx < n_elements:
            # Compute the original index before shifting
            original_index = [global_idx // stride[d] % stride[d] for d in range(num_dims)]
            shifted_index = [(original_index[d] + stride[d] // 2) % stride[d] for d in range(num_dims)]
            shifted_offset = sum(shifted_index[d] * stride[d] for d in range(num_dims))
            o_ptr[global_idx] = x_ptr[shifted_offset]
