import triton
import triton.language as tl

@triton.jit
def div_kernel(input_ptr, other_ptr, output_ptr, N, rounding_mode):
    # Calculate the index for each element
    idx = tl.program_id(0) * tl.num_warps() + tl.warp_id()
    
    # Ensure we do not exceed the bounds
    if idx < N:
        input_val = tl.load(input_ptr + idx)
        other_val = tl.load(other_ptr + idx)
        
        # Perform division
        if other_val == 0:
            result = 0  # Handle division by zero
        else:
            result = input_val / other_val
        
        # Apply rounding if specified
        if rounding_mode == 'floor':
            result = tl.floor(result)
        elif rounding_mode == 'ceil':
            result = tl.ceil(result)
        elif rounding_mode == 'round':
            result = tl.round(result)
        
        # Store the result
        tl.store(output_ptr + idx, result)

def div(input: tl.Tensor, other: tl.Tensor, *, rounding_mode=None, out=None) -> tl.Tensor:
    # Determine the size of the input tensor
    N = input.shape[0]  # Assuming 1D for simplicity; adjust for higher dimensions
    
    # Allocate output tensor if not provided
    if out is None:
        out = tl.empty_like(input)
    
    # Launch the Triton kernel
    div_kernel[(N + 255) // 256](input.data_ptr(), other.data_ptr(), out.data_ptr(), N, rounding_mode)
    
    return out
