import triton.language as tl
import triton

BLOCK_SIZE = 1024

# Triton mul kernel
@triton.jit
def mul2_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    **meta
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    doubled_values = values * 2
    tl.store(output_ptr + offsets, doubled_values, mask=mask)

# Triton mul inplace kernel
@triton.jit
def mul2_inplace_kernel(
    data_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    **meta
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(data_ptr + offsets, mask=mask)
    doubled_values = values * 2
    tl.store(data_ptr + offsets, doubled_values, mask=mask)

def triton_mul2(output_tensor, input_tensor):
    n_elements = input_tensor.numel()
    grid = lambda meta: triton.cdiv(n_elements, BLOCK_SIZE)
    mul2_kernel[grid](output_tensor, input_tensor, n_elements)

def triton_mul2_inplace(input_tensor):
    n_elements = input_tensor.numel()
    grid = lambda meta: triton.cdiv(n_elements, BLOCK_SIZE)
    mul2_inplace_kernel[grid](input_tensor, n_elements)
