import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xb, stride_xh, stride_xl,
    stride_cb, stride_ch, stride_cl,
    stride_sb, stride_sh, stride_sl,
    stride_ob, stride_oh, stride_ol,
    B, H, L, D,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the program ID
    pid = tl.program_id(0)
    
    # Compute the batch and head indices
    b = pid // H
    h = pid % H

    # Compute the memory offsets
    x_offset = b * stride_xb + h * stride_xh
    c_offset = b * stride_cb + h * stride_ch
    s_offset = b * stride_sb + h * stride_sh
    o_offset = b * stride_ob + h * stride_oh

    # Iterate over the sequence length in blocks
    for l in range(0, L, BLOCK_SIZE):
        # Compute the number of elements to process in this block
        N = min(BLOCK_SIZE, L - l)

        # Load the input values
        x = tl.load(X_ptr + x_offset + (l * stride_xl) + tl.arange(0, N)[:, None] * stride_xl + tl.arange(0, D)[None, :])
        
        # Load the cosine and sine values
        cos = tl.load(cos_ptr + c_offset + (l * stride_cl) + tl.arange(0, N)[:, None] * stride_cl + tl.arange(0, D)[None, :])
        sin = tl.load(sin_ptr + s_offset + (l * stride_sl) + tl.arange(0, N)[:, None] * stride_sl + tl.arange(0, D)[None, :])

        # Apply the rotary embedding
        x_rot = tl.where(
            tl.arange(0, D)[None, :] % 2 == 0,
            x * cos - tl.roll(x, 1, 1) * sin,
            x * sin + tl.roll(x, -1, 1) * cos
        )

        # Store the result
        tl.store(out_ptr + o_offset + (l * stride_ol) + tl.arange(0, N)[:, None] * stride_ol + tl.arange(0, D)[None, :], x_rot)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and cos.is_cuda and sin.is_cuda, "All input tensors must be on GPU"
    assert x.dtype == cos.dtype == sin.dtype, "All input tensors must have the same dtype"
    
    B, H, L, D = x.shape
    assert cos.shape == sin.shape == (B, H, L, D), "Shape mismatch between input tensors"

    # Prepare the output tensor
    output = torch.empty_like(x)

    # Define the grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(L)
    grid = (B * H,)

    # Launch the kernel
    rotary_kernel[grid](
        x, cos, sin, output,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        B, H, L, D,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
