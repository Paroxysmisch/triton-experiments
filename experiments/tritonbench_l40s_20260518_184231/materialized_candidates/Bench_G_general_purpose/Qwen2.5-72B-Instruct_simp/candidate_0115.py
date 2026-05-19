import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,  # Pointer to the input tensor
    rms_weights_ptr,  # Pointer to the RMS weights
    out_ptr,  # Pointer to the output tensor
    stride_x_B,  # Stride of x in the B dimension
    stride_x_H,  # Stride of x in the H dimension
    stride_x_K,  # Stride of x in the K dimension
    stride_rms_weights_K,  # Stride of rms_weights in the K dimension
    stride_out_B,  # Stride of out in the B dimension
    stride_out_H,  # Stride of out in the H dimension
    stride_out_K,  # Stride of out in the K dimension
    B,  # Size of the B dimension
    H,  # Size of the H dimension
    K,  # Size of the K dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for the K dimension
):
    # Compute the block index in the B and H dimensions
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Compute the starting index in the B and H dimensions
    start_b = pid_b * BLOCK_SIZE
    start_h = pid_h * BLOCK_SIZE

    # Compute the range of indices in the B and H dimensions
    range_b = tl.arange(0, BLOCK_SIZE) + start_b
    range_h = tl.arange(0, BLOCK_SIZE) + start_h

    # Mask to filter out-of-bounds indices
    mask_b = range_b < B
    mask_h = range_h < H

    # Compute the RMS for each element in the K dimension
    for k in range(K):
        # Load the elements from x
        x = tl.load(x_ptr + range_b[:, None] * stride_x_B + range_h[None, :] * stride_x_H + k * stride_x_K, mask=mask_b[:, None] & mask_h[None, :], other=0.0)

        # Compute the square of the elements
        x_sq = x * x

        # Compute the sum of squares for the current block
        sum_sq = tl.sum(x_sq, axis=1)

        # Compute the RMS for the current block
        rms = tl.sqrt(sum_sq / K)

        # Load the corresponding RMS weights
        rms_weights = tl.load(rms_weights_ptr + k * stride_rms_weights_K)

        # Normalize and scale the elements
        out = x / rms[:, None] * rms_weights

        # Store the normalized and scaled elements in the output tensor
        tl.store(out_ptr + range_b[:, None] * stride_out_B + range_h[None, :] * stride_out_H + k * stride_out_K, out, mask=mask_b[:, None] & mask_h[None, :])

import torch
import triton
import triton.language as tl

def rmsnorm_wrapper(x, rms_weights, out):
    # Ensure the tensors are on the same device
    assert x.device == rms_weights.device == out.device, "Tensors must be on the same device"
    assert x.dtype == rms_weights.dtype == out.dtype, "Tensors must have the same data type"
    assert x.shape == out.shape, "Input and output tensors must have the same shape"

    # Get the dimensions of the input tensor
    B, H, K = x.shape

    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid = (triton.cdiv(B, BLOCK_SIZE), triton.cdiv(H, BLOCK_SIZE))

    # Launch the kernel
    rmsnorm_triton[grid](
        x,  # Pointer to the input tensor
        rms_weights,  # Pointer to the RMS weights
        out,  # Pointer to the output tensor
        x.stride(0),  # Stride of x in the B dimension
        x.stride(1),  # Stride of x in the H dimension
        x.stride(2),  # Stride of x in the K dimension
        rms_weights.stride(0),  # Stride of rms_weights in the K dimension
        out.stride(0),  # Stride of out in the B dimension
        out.stride(1),  # Stride of out in the H dimension
        out.stride(2),  # Stride of out in the K dimension
        B,  # Size of the B dimension
        H,  # Size of the H dimension
        K,  # Size of the K dimension
        BLOCK_SIZE,  # Block size for the K dimension
    )

# Example usage
B, H, K = 16, 32, 64
x = torch.randn(B, H, K, device='cuda')
rms_weights = torch.randn(K, device='cuda')
out = torch.empty_like(x)

rmsnorm_wrapper(x, rms_weights, out)

print(out)
