import triton
import triton.language as tl

@triton.jit
def floor_kernel(input_ptr, output_ptr, n_elements,
                 **meta):
    # Kernel logic for floor operation
    pid = tl.program_id(axis=0)
    block_start = pid * meta['BLOCK_SIZE']
    offsets = block_start + tl.arange(0, meta['BLOCK_SIZE'])
    mask = offsets < n_elements

    input_values = tl.load(input_ptr + offsets, mask=mask)
    output_values = tl.floor(input_values)
    tl.store(output_ptr + offsets, output_values, mask=mask)

def floor(input, *, out=None):
    # Function to call the Triton kernel
    if out is None:
        out = input.new_empty(input.shape)

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    floor_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out
