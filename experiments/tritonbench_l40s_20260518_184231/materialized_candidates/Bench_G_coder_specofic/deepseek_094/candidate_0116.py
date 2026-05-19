import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n):
    # Define your loop
    for i in range(n):
        x = tl.load(x_ptr + i)
        y = tl.load(y_ptr + i)
        z = x + y
        tl.store(z_ptr + i, z)
