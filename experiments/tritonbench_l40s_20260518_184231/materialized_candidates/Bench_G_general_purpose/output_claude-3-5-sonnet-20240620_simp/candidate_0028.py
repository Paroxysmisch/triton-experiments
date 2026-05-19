import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X_ptr, cos_ptr, sin_ptr, output_ptr,
    stride_xb, stride_xh, stride_xm, stride_xl,
    stride_cb, stride_ch, stride_cm,
    stride_sb, stride_sh, stride_sm,
    stride_ob, stride_oh, stride_om, stride_ol,
    B, H, M, L, D, interleaved: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the program ID
    pid = tl.program_id(0)
    
    # Compute the block start indices
    b = pid // (H * M)
    h = (pid % (H * M)) // M
    m = pid % M
    
    # Compute the memory offsets
    x_offset = b * stride_xb + h * stride_xh + m * stride_xm
    c_offset = b * stride_cb + h * stride_ch + m * stride_cm
    s_offset = b * stride_sb + h * stride_sh + m * stride_sm
    o_offset = b * stride_ob + h * stride_oh + m * stride_om
    
    # Load the input data
    x = tl.load(X_ptr + x_offset + tl.arange(0, BLOCK_SIZE) * stride_xl)
    
    # Load the cosine and sine values
    cos = tl.load(cos_ptr + c_offset + tl.arange(0, BLOCK_SIZE))
    sin = tl.load(sin_ptr + s_offset + tl.arange(0, BLOCK_SIZE))
    
    # Perform the rotary position encoding
    if interleaved:
        x_re, x_im = x[0::2], x[1::2]
        output_re = x_re * cos - x_im * sin
        output_im = x_re * sin + x_im * cos
        output = tl.where(tl.arange(0, BLOCK_SIZE) % 2 == 0, output_re, output_im)
    else:
        x_re, x_im = x[:D//2], x[D//2:]
        output_re = x_re * cos - x_im * sin
        output_im = x_re * sin + x_im * cos
        output = tl.cat(output_re, output_im)
    
    # Store the output
    tl.store(output_ptr + o_offset + tl.arange(0, BLOCK_SIZE) * stride_ol, output)

def apply_rotary(x, cos, sin, interleaved=False, conjugate=False):
    assert x.dtype == torch.float32, "Input tensor must be float32"
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32, "Cos and sin tensors must be float32"
    
    B, H, M, L, D = x.shape
    assert cos.shape == (B, H, M, D), "Cos shape mismatch"
    assert sin.shape == (B, H, M, D), "Sin shape mismatch"
    
    output = torch.empty_like(x)
    
    if conjugate:
        sin = -sin
    
    BLOCK_SIZE = D
    grid = (B * H * M,)
    
    rotary_kernel[grid](
        x, cos, sin, output,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, M, L, D, interleaved,
        BLOCK_SIZE
    )
    
    return output
