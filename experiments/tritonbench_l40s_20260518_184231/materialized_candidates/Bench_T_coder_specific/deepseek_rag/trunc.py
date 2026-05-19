import triton
import triton.language as tl

@triton.jit
def truncate_kernel(input_ptr, output_ptr, N):
    pid = tl.program_id(0)
    if pid < N:
        input_val = tl.load(input_ptr + pid)
        output_val = tl.trunc(input_val)
        tl.store(output_ptr + pid, output_val)
