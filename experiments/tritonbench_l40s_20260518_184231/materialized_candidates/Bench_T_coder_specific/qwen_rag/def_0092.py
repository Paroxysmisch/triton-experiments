import triton
import triton.language as tl
import torch

@triton.jit
def tensordot_rsqrt_kernel(a_ptr, b_ptr, c_ptr, a_shape, b_shape, result_shape, strides_a, strides_b, strides_c, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    coords = tl.unravel(pid, (num_elements,))
    
    acc = 0.0
    for k in range(BLOCK_SIZE):
        idx_a = [coords[i] // strides_a[i] for i in range(len(coords))]
        idx_b = [coords[i] // strides_b[i] for i in range(len(coords))]
        acc += a_ptr[tl.linear(idx_a)] * b_ptr[tl.linear(idx_b)]
    
    acc = tl.reduce(acc, axis=0, op=tl.sum)
    out_val = tl.rsqrt(acc)
    tl.store(c_ptr + coords[0], out_val)

# Wrapper function to launch the Triton kernel
def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims):
    assert isinstance(dims, tuple) and len(dims) == 2, "dims must be a tuple of two lists"
    
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    result_shape = []
    
    # Compute the result shape after contraction
    for i in range(len(a_shape)):
        if i < len(dims[0]):
            continue
        result_shape.append(a_shape[i])
    for i in range(len(b_shape)):
        if i < len(dims[1]):
            continue
        result_shape.append(b_shape[i])
    
    # Flatten the shapes and strides for easy indexing
    flat_a_shape = [1] + a_shape[1:]
    flat_b_shape = [1] + b_shape[1:]
    flat_result_shape = [1] + result_shape[1:]
    
    strides_a = [flat_a_shape[i+1:].product() for i in range(len(flat_a_shape)-1)]
    strides_b = [flat_b_shape[i+1:].product() for i in range(len(flat_b_shape)-1)]
    strides_c = [flat_result_shape[i+1:].product() for i in range(len(flat_result_shape)-1)]
    
    num_elements = result_shape[0]
    block_size = min(num_elements, 256)
    grid_size = (num_elements + block_size - 1) // block_size
    
    c = torch.zeros(result_shape, dtype=a.dtype, device=a.device)
    
    tensordot_rsqrt_kernel[(grid_size,)](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=c.data_ptr(),
        a_shape=a_shape,
        b_shape=b_shape,
        result_shape=result_shape,
        strides_a=strides_a,
        strides_b=strides_b,
        strides_c=strides_c,
        num_elements=num_elements,
        BLOCK_SIZE=block_size
    )
    
    return c
