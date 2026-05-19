import triton
import triton.language as tl

@triton.jit
def bessel_j1_kernel(input_ptr, output_ptr, n):
    pid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    num_warps = 4
    num_threads_per_block = 256
    block_size = num_threads_per_block * num_warps
    
    i = pid * block_size + tid
    if i < n:
        x = input_ptr[i]
        y = tl.besselj1(x)
        output_ptr[i] = y
