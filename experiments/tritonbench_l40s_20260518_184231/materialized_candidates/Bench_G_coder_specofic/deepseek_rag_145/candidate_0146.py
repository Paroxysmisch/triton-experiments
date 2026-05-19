import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    out_ptr, out_stride,
    in_ptr, in_stride,
    bias_ptr, bias_stride,
    scale, activation,
    BLOCK_SIZE: tl.constexpr,
):
    # Get our thread index
    pid = tl.program_id(axis=0)
    # Get our matrix index
    idx = pid
    
    # Initialize the output as the bias
    out = bias_ptr[idx]
    
    # Perform the fused operation
    for block_start in range(0, in_stride, BLOCK_SIZE):
        # Load the block of data
        x = in_ptr[block_start + idx]
        
        # Add and scale
        out += scale * x

    # Apply the activation function
    if activation == "sigmoid":
        out = 1.0 / (1.0 + tl.exp(-out))
    elif activation == "relu":
        out = tl.maximum(out, 0.0)

    # Store the result
    out_ptr[idx] = out
