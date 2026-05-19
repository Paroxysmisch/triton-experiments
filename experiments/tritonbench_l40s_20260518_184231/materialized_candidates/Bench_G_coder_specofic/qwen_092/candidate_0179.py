import triton
import triton.language as tl

@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + block_start + offsets, mask=block_start + offsets < n_elements)
    absmax_inv = tl.load(absmax_inv_ptr)
    scaled_values = x * absmax_inv
    quantized_values = tl.extra.cuda.libdevice.llrint(scaled_values)
    tl.store(output_ptr + block_start + offsets, quantized_values, mask=block_start + offsets < n_elements)
