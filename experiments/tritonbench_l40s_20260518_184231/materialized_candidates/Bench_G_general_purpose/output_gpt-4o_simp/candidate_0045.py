import triton
import triton.language as tl

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    val0, in0_ptr, out0_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the index of the current thread
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle out-of-bounds
    mask = offsets < n_elements

    # Load input tensor elements
    in0 = tl.load(in0_ptr + offsets, mask=mask)

    # Perform element-wise exponentiation
    out0 = tl.math.pow(val0, in0)

    # Store the result
    tl.store(out0_ptr + offsets, out0, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0):
    # Determine the number of elements
    n_elements = in0.size

    # Define block size heuristically
    BLOCK_SIZE = 1024  # This can be tuned based on the hardware and input size

    # Calculate the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        val0, in0, out0, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage:
# import torch
# val0 = 2.0  # Scalar value
# in0 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')  # Input tensor
# out0 = torch.empty_like(in0)  # Output tensor
# pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0)
# print(out0)  # Should print tensor([2.0, 4.0, 8.0, 16.0], device='cuda')
