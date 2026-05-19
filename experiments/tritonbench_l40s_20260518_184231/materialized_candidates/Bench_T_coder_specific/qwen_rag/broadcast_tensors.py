import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(
    output_ptr,
    input_ptr,
    input_shape,
    output_shape,
    stride_input,
    stride_output,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(num_elements, BLOCK_SIZE)
    
    # Calculate indices
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = idx % num_elements
    
    # Unpack input shape
    input_shape = tl.tensor(input_shape, dtype=tl.int32)
    output_shape = tl.tensor(output_shape, dtype=tl.int32)
    
    # Calculate output indices
    output_idx = idx
    for i in range(len(output_shape)):
        dim_size = output_shape[i]
        if dim_size != input_shape[i]:
            dim_ratio = dim_size // input_shape[i]
            output_idx //= dim_ratio
    
    # Load input element
    input_val = tl.load(input_ptr + stride_input * output_idx)
    
    # Store output element
    tl.store(output_ptr + stride_output * idx, input_val)
