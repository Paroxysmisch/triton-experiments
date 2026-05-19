import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,
    bias_ptr,
    in_ptr,
    in_out_ptr,
    scale_ptr,
    bias_ld,
    in_ld,
    in_out_ld,
    N,
    K,
    ACTIVATION_TYPE: tl.constexpr,
    MULTIPLIER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program id
    pid = tl.program_id(0)
    # Compute offsets for block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load bias
    bias_ptr += offsets
    bias = tl.load(bias_ptr)
    # Compute offsets for block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for bounds checking
    mask = offsets < N * K
    # Initialize pointers
    x_ptr = x_ptr + offsets
    in_ptr = in_ptr + offsets
    in_out_ptr = in_out_ptr + offsets
    # Load x
    x = tl.load(x_ptr, mask=mask)
    # Broadcast bias
    bias = tl.broadcast(bias, BLOCK_SIZE)
    # Compute offsets for block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Create mask for bounds checking
    mask = offsets < K
    # Load scale
    scale_ptr += offsets
    scale = tl.load(scale_ptr, mask=mask)
    # Compute in
    in_ = x + bias
    in_ = in_ * scale
    # Store in_out
    tl.store(in_out_ptr, in_, mask=mask)
    # Compute out
    out = tl.where(ACTIVATION_TYPE == "sigmoid", tl.sigmoid(in_ * MULTIPLIER), in_)
    out = tl.where(ACTIVATION_TYPE == "relu", tl.maximum(out, 0), out)
    # Store out
    tl.store(in_out_ptr, out, mask=mask)

def fused_add_mul_activation_torch(
    in_out_tensor,
    bias,
    in_tensor,
    activation="sigmoid",
    multiplier=1.0,
    max_grid=1024,
    eviction_policy="lru",
):
    # Define constants
    BLOCK_SIZE = 128
    # Normalize multiplier
    multiplier = max(multiplier, 1e-3)
    # Get size parameters
    N, K = in_out_tensor.shape
    # Define grid
    grid = lambda meta: (triton.cdiv(N * K, BLOCK_SIZE),)
    # Ensure dtype support
    assert in_out_tensor.dtype in [torch.float16, torch.bfloat16]
    assert in_tensor.dtype in [torch.float16, torch.bfloat16]
    assert bias.dtype in [torch.float16, torch.bfloat16]
    # Cast input tensors to fp16 if not already
    if in_out_tensor.dtype != torch.float16:
        in_out_tensor = in_out_tensor.to(torch.float16)
        in_tensor = in_tensor.to(torch.float16)
    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)
    # Call kernel
    fused_add_mul_activation_kernel[grid](
        in_out_tensor,
        bias,
        in_tensor,
        in_out_tensor,
        scale_ptr=torch.tensor(1.0 / multiplier, dtype=torch.float16),
        bias_ld=bias.stride(0),
        in_ld=in_tensor.stride(0),
        in_out_ld=in_out_tensor.stride(0),
        N=N,
        K=K,
        ACTIVATION_TYPE=activation,
        MULTIPLIER=multiplier,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
        eviction_policy=eviction_policy,
        max_grid=(max_grid,),
    )
    return in_out_tensor
