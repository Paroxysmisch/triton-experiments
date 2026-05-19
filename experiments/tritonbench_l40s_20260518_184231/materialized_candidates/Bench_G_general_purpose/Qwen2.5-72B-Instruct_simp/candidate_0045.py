import triton
import triton.language as tl

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    val0: tl.float32,
    in0_ptr: tl.tensor,
    out0_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in0_ptr + offsets, mask=mask)
    y = tl.math.pow(x, val0)
    tl.store(out0_ptr + offsets, y, mask=mask)

import triton
import triton.runtime as tr
import triton.language as tl

def pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0, BLOCK_SIZE=128, num_warps=4):
    # Get the number of elements in the tensor
    n_elements = in0.size

    # Determine the grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[grid_size, BLOCK_SIZE, num_warps](
        val0,
        in0.ptr,
        out0.ptr,
        n_elements,
        BLOCK_SIZE
    )

import torch
import triton

# Example input tensor
in0 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
# Example scalar value
val0 = 2.0
# Output tensor
out0 = torch.empty_like(in0)

# Launch the wrapper
pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0)

# Print the result
print(out0)  # Output: tensor([1., 4., 9., 16.])
