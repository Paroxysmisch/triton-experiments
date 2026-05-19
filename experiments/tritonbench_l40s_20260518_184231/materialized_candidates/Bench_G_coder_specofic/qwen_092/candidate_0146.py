import torch
import triton
import triton.language as tl

# Triton kernel signature
@triton.jit
def fused_add_mul_activation_kernel(
    out_ptr: tl.tensor,
    in_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    multiplier_ptr: tl.tensor,
    N: tl.int32,
    scale: tl.float32,
    multiplier: tl.float32,
    activation: tl.char_ptr,
):
    idx = tl.program_id(axis=0)
    block_start = idx * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)

    for i in range(block_start, block_end):
        x = in_ptr[i]
        bias = bias_ptr[i]
        scale_x = scale * x
        result = x + bias + scale_x * multiplier

        if activation[0] == b's':  # Sigmoid activation
            result = 1.0 / (1.0 + tl.exp(-result))
        elif activation[0] == b'r':  # ReLU activation
            result = result if result > 0 else 0

        out_ptr[i] = result

# PyTorch wrapper function
def fused_add_mul_activation_torch(
    in_out_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    multiplier_tensor: torch.Tensor,
    scale: float = 1.0,
    multiplier: float = 1.0,
    activation: str = 'sigmoid'
):
    # Ensure the input tensors are on the GPU
    in_out_tensor = in_out_tensor.cuda()
    bias_tensor = bias_tensor.cuda()
    scale_tensor = scale_tensor.cuda()
    multiplier_tensor = multiplier_tensor.cuda()

    # Get the size of the input tensor
    N = in_out_tensor.size(0)

    # Create output tensor on the same device as input tensor
    out_tensor = torch.empty_like(in_out_tensor)

    # Configure the grid and block size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    block = (BLOCK_SIZE,)

    # Launch the Triton kernel
    fused_add_mul_activation_kernel[grid, block](
        out_tensor,
        in_out_tensor,
        bias_tensor,
        scale_tensor,
        multiplier_tensor,
        N,
        scale,
        multiplier,
        activation.encode('utf-8')
    )

    return out_tensor
