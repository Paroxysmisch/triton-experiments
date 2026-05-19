import triton
import triton.language as tl

@triton.jit
def exp_kernel(input_ptr, output_ptr, n):
    tid = tl.program_id(0)
    stride = tl.cuda.grid_stride_loop(n)
    while tid < n:
        output_ptr[tid] = math.exp(input_ptr[tid])
        tid += stride

def exp(input_tensor, out=None):
    if out is None:
        out = triton.empty_like(input_tensor)
    else:
        if not triton.is_compatible(out, input_tensor):
            raise ValueError("out tensor is not compatible with input tensor")

    grid = lambda : (out.numel+63)//64*64
    exp_kernel[grid](input_tensor, out, out.numel)

    return out
