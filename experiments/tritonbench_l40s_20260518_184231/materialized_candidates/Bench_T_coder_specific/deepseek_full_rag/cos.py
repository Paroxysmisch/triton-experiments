import triton
import triton.language as tl

@triton.jit
def cos(x, out=None):
    # Compute the cosine of elements
    return tl.cos(x)

def cos(input, *, out=None):
    # Convert input to a Triton tensor
    input_triton = triton.language.promote_to_tensor(input)
    # Call the Triton function to compute the cosine
    output_triton = cos(input_triton, out=out)
    # Convert the Triton tensor back to a standard Python tensor
    output = output_triton.to_torch()
    return output
