import triton
import triton.language as tl
import torch

# Triton kernel for RMS normalization
@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   x_strides: tl.int32,
                   rms_w_strides: tl.int32,
                   output_ptr: tl.pointer_type,
                   output_strides: tl.int32,
                   N_SIZE: tl.int32,
                   eps: tl.float32,
                   BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Calculate the starting pointer for the current batch and row
    x_start_ptr = x_ptr + pid_batch * x_strides + pid_m * N_SIZE
    rms_w_start_ptr = rms_w_ptr + pid_batch * rms_w_strides
    output_start_ptr = output_ptr + pid_batch * output_strides + pid_m * N_SIZE

    # Iterate over chunks of size BLOCK_N_SIZE
    for chunk_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offsets = chunk_idx + tl.arange(0, BLOCK_N_SIZE)
        mask = offsets < N_SIZE

        # Load input and weight
        x_chunk = tl.load(x_start_ptr + offsets, mask=mask, other=0)
        rms_w_chunk = tl.load(rms_w_start_ptr + offsets, mask=mask, other=1)

        # Compute the sum of squares
        squared_chunk = x_chunk * x_chunk
        sum_squares = tl.sum(squared_chunk, axis=0)
        variance = sum_squares / N_SIZE
        rstd = 1.0 / tl.sqrt(variance + eps)

        # Normalize and apply weight
        normalized_chunk = x_chunk * rstd
        output_chunk = normalized_chunk * rms_w_chunk

        # Store the result
        tl.store(output_start_ptr + offsets, output_chunk, mask=mask)

# Wrapper function for RMS normalization
def rmsnorm_triton_wrapper(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6):
    assert x.is_cuda and rms_w.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert rms_w.is_contiguous(), "Weight tensor must be contiguous"
    assert x.dim() == 3, "Input tensor must be 3D"
    assert rms_w.dim() == 1, "Weight tensor must be 1D"
    assert rms_w.shape[0] == x.shape[-1], "Weight tensor size must match the last dimension of input tensor"

    B, M, N = x.shape
    output = torch.empty_like(x)

    # Define the grid and block dimensions
    grid = (B, M)
    BLOCK_N_SIZE = triton.next_power_of_2(N)

    # Launch the kernel
    rmsnorm_triton[grid](
        x, rms_w, x.stride(0) * x.stride(1), rms_w.stride(0), output, output.stride(0) * output.stride(1),
        N, eps, BLOCK_N_SIZE,
        num_warps=4
    )

    return output
