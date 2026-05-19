import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_weights_ptr: tl.pointer_type,
                   out_ptr: tl.pointer_type,
                   K: tl.uint32,
                   eps: tl.float32,
                   BLOCK_SIZE: tl.constexpr):
    # Compute the row index for this instance
    row_idx = tl.program_id(0)
    # Calculate the start pointer for the current row
    row_start_ptr = x_ptr + row_idx * K
    # Define offsets for loading data
    offsets = tl.arange(0, BLOCK_SIZE)
    # Mask for bounds checking
    mask = offsets < K

    # Load the current row and rms_weights
    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    rms_weights = tl.load(rms_weights_ptr + offsets, mask=mask, other=1)

    # Compute RMS
    squared_row = x_row * x_row
    squared_mean = tl.sum(squared_row) / K
    rms = tl.sqrt(squared_mean + eps)

    # Normalize and scale by rms_weights
    normalized_row = x_row / rms
    scaled_row = normalized_row * rms_weights

    # Store the result in the output
    tl.store(out_ptr + row_idx * K + offsets, scaled_row, mask=mask)

def rmsnorm_wrapper(x, rms_weights, eps=1e-5):
    assert x.is_cuda and rms_weights.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert rms_weights.is_contiguous(), "Weights tensor must be contiguous"
    
    # Get dimensions
    B, N, K = x.shape
    BLOCK_SIZE = triton.next_power_of_2(K)

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch the kernel
    rmsnorm_triton[(B * N,)](
        x, rms_weights, out, K, eps,
        num_warps=16, BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Example usage
B, N, K = 4, 8, 16  # Example dimensions
x = torch.randn(B, N, K, device='cuda')
rms_weights = torch.ones(K, device='cuda')  # Example weights
out = rmsnorm_wrapper(x, rms_weights)
print(out)
