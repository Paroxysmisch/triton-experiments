import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, N):
    pid = tl.program_id(0)
    block_start = pid * tl.program_id(1)
    offsets = tl.arange(0, N)
    mask = block_start + offsets < N
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(output_ptr + offsets, y, mask=mask)

def relu(input_tensor, output_tensor):
    N = input_tensor.numel()
    num_blocks = triton.next_power_of_2(N) // 256
    input_ptr = triton.pointers.register_buffer(input_tensor)
    output_ptr = triton.pointers.register_buffer(output_tensor)
    relu_kernel[num_blocks, 256](input_ptr, output_ptr, N)
    return output_tensor
