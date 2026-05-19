import triton
import triton.language as tl

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    val0: tl.float32,  # Scalar value
    in0_ptr: tl.tensor,  # Pointer to the input tensor
    out0_ptr: tl.tensor,  # Pointer to the output tensor
    n_elements: tl.int32,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the start index for this block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for this block
    mask = offsets < n_elements  # Create a mask to ensure we don't read out of bounds

    in0 = tl.load(in0_ptr + offsets, mask=mask)  # Load the input tensor elements
    out0 = tl.math.pow(in0, val0)  # Perform the power operation
    tl.store(out0_ptr + offsets, out0, mask=mask)  # Store the result in the output tensor

import torch

def pow_func_scalar_tensor_wrapper_rank_1(
    val0: float,
    in0: torch.Tensor,
    out0: torch.Tensor
):
    # Ensure the input and output tensors are on the same device
    assert in0.device == out0.device, "Input and output tensors must be on the same device"
    assert in0.dtype == out0.dtype, "Input and output tensors must have the same data type"
    assert in0.dim() == 1, "Input tensor must be a 1D tensor"
    assert out0.dim() == 1, "Output tensor must be a 1D tensor"
    assert in0.size(0) == out0.size(0), "Input and output tensors must have the same size"

    n_elements = in0.numel()  # Number of elements in the tensor

    # Determine the optimal block size
    BLOCK_SIZE = 1024  # This can be adjusted based on the GPU architecture and performance tuning

    # Determine the grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[grid_size, BLOCK_SIZE](
        val0,
        in0,
        out0,
        n_elements,
        BLOCK_SIZE
    )
