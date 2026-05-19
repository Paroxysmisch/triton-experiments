import triton
import triton.language as tl

@triton.jit
def conv2d_forward_kernel(
    input_ptr,  # Pointer to the input tensor
    weight_ptr,  # Pointer to the weight tensor
    output_ptr,  # Pointer to the output tensor
    N,  # Batch size
    C,  # Number of input channels
    H,  # Input height
    W,  # Input width
    K,  # Number of output channels
    R,  # Kernel height
    S,  # Kernel width
    P,  # Padding height
    Q,  # Padding width
    stride_h,  # Stride height
    stride_w,  # Stride width
    groups,  # Number of groups
    input_stride_N,  # Stride of input tensor in batch dimension
    input_stride_C,  # Stride of input tensor in channel dimension
    input_stride_H,  # Stride of input tensor in height dimension
    input_stride_W,  # Stride of input tensor in width dimension
    weight_stride_K,  # Stride of weight tensor in output channel dimension
    weight_stride_C,  # Stride of weight tensor in input channel dimension
    weight_stride_R,  # Stride of weight tensor in height dimension
    weight_stride_S,  # Stride of weight tensor in width dimension
    output_stride_N,  # Stride of output tensor in batch dimension
    output_stride_K,  # Stride of output tensor in output channel dimension
    output_stride_P,  # Stride of output tensor in height dimension
    output_stride_Q,  # Stride of output tensor in width dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for output channel dimension
    BLOCK_SIZE_P: tl.constexpr,  # Block size for height dimension
    BLOCK_SIZE_Q: tl.constexpr,  # Block size for width dimension
    BLOCK_SIZE_R: tl.constexpr,  # Block size for kernel height dimension
    BLOCK_SIZE_S: tl.constexpr,  # Block size for kernel width dimension
):
    # Get the program ID
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_p = tl.program_id(axis=2)
    pid_q = tl.program_id(axis=3)

    # Compute the global indices
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # Mask to avoid out-of-bounds access
    mask_n = n < N
    mask_k = k < K
    mask_p = p < P
    mask_q = q < Q

    # Compute the input and output indices
    input_indices = n[:, None, None, None] * input_stride_N + \
                    (k // (K // C))[:, None, None] * input_stride_C + \
                    (p * stride_h + tl.arange(0, BLOCK_SIZE_R))[:, None] * input_stride_H + \
                    (q * stride_w + tl.arange(0, BLOCK_SIZE_S)) * input_stride_W

    output_indices = n[:, None, None, None] * output_stride_N + \
                     k[:, None, None] * output_stride_K + \
                     p[:, None] * output_stride_P + \
                     q * output_stride_Q

    # Load the input and weight data
    input_data = tl.load(input_ptr + input_indices, mask=mask_n[:, None, None, None] & mask_k[:, None, None] & mask_p[:, None] & mask_q, other=0.0)
    weight_data = tl.load(weight_ptr + k[:, None, None, None] * weight_stride_K + \
                          (k // (K // C))[:, None, None] * weight_stride_C + \
                          tl.arange(0, BLOCK_SIZE_R)[:, None] * weight_stride_R + \
                          tl.arange(0, BLOCK_SIZE_S) * weight_stride_S, mask=mask_k[:, None, None, None] & mask_p[:, None] & mask_q, other=0.0)

    # Compute the output
    output_data = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_P, BLOCK_SIZE_Q), dtype=tl.float32)
    for r in range(R):
        for s in range(S):
            output_data += input_data[:, :, r, s, None] * weight_data[:, :, r, s, None]

    # Store the output
    tl.store(output_ptr + output_indices, output_data, mask=mask_n[:, None, None, None] & mask_k[:, None, None] & mask_p[:, None] & mask_q)

import torch
import triton
import triton.language as tl

def conv2d_forward(input: torch.Tensor, weight: torch.Tensor, stride: Tuple[int, int], padding: Tuple[int, int], groups: int = 1):
    # Get the dimensions
    N, C, H, W = input.shape
    K, Cg, R, S = weight.shape
    assert C % groups == 0, "Input channels must be divisible by the number of groups"
    assert K % groups == 0, "Output channels must be divisible by the number of groups"
    Cg = C // groups

    # Compute the output dimensions
    P = (H + 2 * padding[0] - R) // stride[0] + 1
    Q = (W + 2 * padding[1] - S) // stride[1] + 1

    # Initialize the output tensor
    output = torch.empty((N, K, P, Q), device=input.device, dtype=input.dtype)

    # Define block sizes
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_P = 16
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_R = R
    BLOCK_SIZE_S = S

    # Define grid dimensions
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(K, BLOCK_SIZE_K), triton.cdiv(P, BLOCK_SIZE_P), triton.cdiv(Q, BLOCK_SIZE_Q))

    # Launch the kernel
    conv2d_forward_kernel[grid](
        input, weight, output,
        N, C, H, W, K, R, S, P, Q,
        stride[0], stride[1],
        groups,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_P, BLOCK_SIZE_Q, BLOCK_SIZE_R, BLOCK_SIZE_S
    )

    return output
