import triton
import triton.language as tl
import torch
from typing import Union, Tuple

@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    B, C, H_in, W_in, H_out, W_out,
    stride_b_in, stride_c_in, stride_h_in, stride_w_in,
    stride_b_out, stride_c_out, stride_h_out, stride_w_out,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Compute row and column indices in the output.
    out_h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Limit to valid output range.
    mask_h = out_h < H_out
    mask_w = out_w < W_out

    b = tl.program_id(2)
    c = tl.program_id(3)

    # Expand indices to 2D meshgrid for h, w
    out_h = out_h[:, None]
    out_w = out_w[None, :]

    # Where valid (mask), calculate the input region for adaptive pooling
    start_h = (out_h * H_in) // H_out
    end_h   = ((out_h + 1) * H_in) // H_out
    start_w = (out_w * W_in) // W_out
    end_w   = ((out_w + 1) * W_in) // W_out

    # Gather pooling region bounds
    sh, eh = start_h, end_h
    sw, ew = start_w, end_w
    
    # Initialize output sum
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over the pooling window
    for hh in range(0, tl.max(eh - sh)):
        cur_h = sh + hh
        # Guard
        valid_h = cur_h < eh
        for ww in range(0, tl.max(ew - sw)):
            cur_w = sw + ww
            valid_w = cur_w < ew
            # Load input if valid
            in_h = tl.where(valid_h, cur_h, 0)
            in_w = tl.where(valid_w, cur_w, 0)

            in_offset = b * stride_b_in \
                        + c * stride_c_in \
                        + in_h.squeeze(-1) * stride_h_in \
                        + in_w.squeeze(0) * stride_w_in
            # For positions outside of region, we assign 0
            val = tl.where(valid_h & valid_w & mask_h & mask_w,
                           tl.load(input_ptr + in_offset, mask=True, other=0.0),
                           0.0)
            acc += val
    
    # Calculate area per output element
    pool_h = (end_h - start_h).to(tl.float32)
    pool_w = (end_w - start_w).to(tl.float32)
    area   = pool_h * pool_w

    # Avoid division by zero
    area = tl.where(area == 0, 1.0, area)

    # Average pooling result
    avg_val = acc / area

    # Apply sigmoid
    sig_res = 1.0 / (1.0 + tl.exp(-avg_val))

    # Store result back to output if valid
    out_offset = b * stride_b_out \
                 + c * stride_c_out \
                 + out_h.squeeze(-1) * stride_h_out \
                 + out_w.squeeze(0) * stride_w_out
    tl.store(output_ptr + out_offset, sig_res, mask=mask_h[:, None] & mask_w[None, :])

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Expect input in shape [B, C, H, W]
    B, C, H_in, W_in = input.shape
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    else:
        H_out, W_out = output_size

    # Create output tensor
    output = torch.empty((B, C, H_out, W_out), device=input.device, dtype=input.dtype)

    # Strides
    stride_b_in, stride_c_in, stride_h_in, stride_w_in = input.stride()
    stride_b_out, stride_c_out, stride_h_out, stride_w_out = output.stride()

    BLOCK_H = 8
    BLOCK_W = 8

    grid = (
        ( (H_out + BLOCK_H - 1) // BLOCK_H ),
        ( (W_out + BLOCK_W - 1) // BLOCK_W ),
        B,
        C
    )
    
    _sigmoid_adaptive_avg_pool2d_kernel[grid](
        input.data_ptr(), output.data_ptr(),
        B, C, H_in, W_in, H_out, W_out,
        stride_b_in, stride_c_in, stride_h_in, stride_w_in,
        stride_b_out, stride_c_out, stride_h_out, stride_w_out,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )

    return output
