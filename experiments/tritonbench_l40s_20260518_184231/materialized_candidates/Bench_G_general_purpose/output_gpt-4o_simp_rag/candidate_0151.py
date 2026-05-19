import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   output_ptr: tl.pointer_type,
                   B: tl.uint32,  # Batch size
                   M: tl.uint32,  # M dimension size
                   BLOCK_N_SIZE: tl.constexpr,  # Block size
                   eps: tl.float32):
    # Get the current program ID, which corresponds to the batch index.
    batch_idx = tl.program_id(0)
    # Compute the start pointers for the current batch.
    x_start_ptr = x_ptr + batch_idx * M
    output_start_ptr = output_ptr + batch_idx * M

    # Compute the offsets within a block.
    offsets = tl.arange(0, BLOCK_N_SIZE)
    mask = offsets < M

    # Load the input data and weights.
    x_row = tl.load(x_start_ptr + offsets, mask=mask, other=0)
    rms_w = tl.load(rms_w_ptr + offsets, mask=mask, other=1)

    # Compute the squared mean and RMS.
    squared_row = x_row * x_row
    squared_mean = tl.sum(squared_row, axis=0) / M
    rms = tl.sqrt(squared_mean + eps)

    # Normalize and scale by the weight vector.
    normalized_row = x_row / rms
    scaled_row = normalized_row * rms_w

    # Store the result in the output.
    tl.store(output_start_ptr + offsets, scaled_row, mask=mask)

def rmsnorm_triton_wrapper(x, rms_w, BLOCK_N_SIZE):
    # Check the input tensor dimensions.
    assert x.ndim == 3, "Input tensor must be 3D"
    B, _, M = x.shape
    assert rms_w.shape[0] == M, "Weight vector size must match the M dimension of the input tensor"
    assert x.is_cuda and rms_w.is_cuda, "Input tensors must be on CUDA"

    # Prepare the output tensor.
    output = torch.empty_like(x)

    # Launch the kernel with B instances in the grid.
    rmsnorm_triton[(B,)](
        x, rms_w, output,
        B, M, BLOCK_N_SIZE,
        eps=1e-5,  # Epsilon for numerical stability
        num_warps=4  # Number of warps to use
    )
    return output

# Example usage:
# x = torch.randn(32, 128, 64, device='cuda')  # Example 3D tensor
# rms_w = torch.randn(64, device='cuda')  # Example weight vector
# BLOCK_N_SIZE = 64  # Define block size
# output = rmsnorm_triton_wrapper(x, rms_w, BLOCK_N_SIZE)
