import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X,  # Input tensor
    cos,  # Cosine values
    sin,  # Sine values
    rotary_dim,  # Dimension to apply rotary embeddings
    seq_len,  # Sequence length
    stride,  # Stride for the input tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    x_ptrs = X + offsets * stride
    cos_ptrs = cos + offsets * stride
    sin_ptrs = sin + offsets * stride

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptrs, mask=mask, other=1.0)
    sin_val = tl.load(sin_ptrs, mask=mask, other=0.0)

    # Compute the rotary embeddings
    x_rot = x * cos_val + tl.flip(x, 1) * sin_val

    tl.store(x_ptrs, x_rot, mask=mask)

import torch
import triton
import triton.language as tl

def apply_rotary(X, cos, sin, rotary_dim, seq_len, block_size=128):
    # Ensure the input tensors are on the same device
    assert X.device == cos.device == sin.device, "Input tensors must be on the same device"
    device = X.device

    # Ensure the input tensors have the correct shape
    assert X.shape[1] == cos.shape[1] == sin.shape[1], "Input tensors must have the same sequence length"
    assert X.shape[2] == cos.shape[2] == sin.shape[2], "Input tensors must have the same rotary dimension"

    # Compute the number of blocks
    num_blocks = (seq_len + block_size - 1) // block_size

    # Define the grid and block configuration
    grid = (num_blocks,)

    # Launch the Triton kernel
    rotary_kernel[grid](
        X, cos, sin, rotary_dim, seq_len, X.stride(0), BLOCK_SIZE=block_size
    )

# Example usage
if __name__ == "__main__":
    # Example input tensor
    X = torch.randn(1, 128, 64, device='cuda')
    seq_len = X.shape[1]
    rotary_dim = X.shape[2]

    # Example cosine and sine values
    cos = torch.cos(torch.linspace(0, 1, rotary_dim, device='cuda')).unsqueeze(0).expand(1, seq_len, -1)
    sin = torch.sin(torch.linspace(0, 1, rotary_dim, device='cuda')).unsqueeze(0).expand(1, seq_len, -1)

    # Apply rotary positional embeddings
    apply_rotary(X, cos, sin, rotary_dim, seq_len)

    print(X)
