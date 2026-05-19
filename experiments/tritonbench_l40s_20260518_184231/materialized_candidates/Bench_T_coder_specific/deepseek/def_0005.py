import triton
import triton.language as tl

@triton.jit
def relu_sqrt(input, inplace=False, out=None):
    # Get the index of the current thread
    idx = tl.program_id(axis=0)
    
    # Load the input value
    val = input[idx] if not inplace else input
    
    # Apply the ReLU function
    val = tl.max(val, 0)
    
    # Apply the square root function
    val = tl.sqrt(val)
    
    # Store the output value
    if out is not None:
        out[idx] = val
    elif inplace:
        input[idx] = val
    
    return out
