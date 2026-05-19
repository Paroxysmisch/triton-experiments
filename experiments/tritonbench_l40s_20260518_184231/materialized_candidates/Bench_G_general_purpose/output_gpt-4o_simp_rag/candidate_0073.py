import triton
import triton.language as tl
import torch

# Triton kernel to calculate the maximum along the second dimension
@triton.jit
def load_reduce_kernel(
    input_ptr, output_ptr,
    stride_in_m, stride_in_n,
    stride_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Define program ID and range
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)

    # Load input data
    input_offsets = offsets_m[:, None] * stride_in_m + offsets_n[None, :] * stride_in_n
    input_data = tl.load(input_ptr + input_offsets, mask=(offsets_m[:, None] < stride_in_m))

    # Compute maximum along the second dimension
    max_values = tl.max(input_data, axis=1)

    # Store result in output
    output_offsets = pid_m * stride_out + offsets_m
    tl.store(output_ptr + output_offsets, max_values, mask=(offsets_m < stride_in_m))

# Python wrapper to test the Triton kernel
def load_reduce(input_tensor, block_m, block_n):
    # Ensure input tensor is contiguous
    input_tensor = input_tensor.contiguous()
    # Prepare output tensor
    output_tensor = torch.empty(input_tensor.size(0), device=input_tensor.device, dtype=input_tensor.dtype)

    # Get strides
    stride_in_m, stride_in_n = input_tensor.stride()
    stride_out = output_tensor.stride(0)

    # Launch the Triton kernel
    grid = (triton.cdiv(input_tensor.size(0), block_m),)
    load_reduce_kernel[grid](
        input_tensor, output_tensor,
        stride_in_m, stride_in_n,
        stride_out,
        BLOCK_M=block_m, BLOCK_N=block_n
    )

    return output_tensor

# Example usage
if __name__ == "__main__":
    BLOCK_M = 128
    BLOCK_N = 128
    input_matrix = torch.randn(1024, 1024, device='cuda')
    result = load_reduce(input_matrix, BLOCK_M, BLOCK_N)
    print(result)
