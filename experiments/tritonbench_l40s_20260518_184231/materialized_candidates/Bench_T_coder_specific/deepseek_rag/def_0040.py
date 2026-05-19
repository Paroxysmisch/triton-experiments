import triton
import triton.language as tl

@triton.jit
def my_function(input, output):
    output[...] = input[...] * 2
